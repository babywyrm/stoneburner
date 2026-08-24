"""Async runners used by the API server routes."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from atomics.api.job_progress import (
    EvalJobReporter,
    FixtureRow,
    LoadJobReporter,
    burn_row,
    eval_fixture_total,
    fixture_row,
    select_eval_fixtures,
)
from atomics.api.jobs import Job
from atomics.api.models import EvalRequest, RunRequest
from atomics.config import load_settings
from atomics.eval.adversarial.runner import run_adversarial
from atomics.eval.budget import EvalBudget, EvalBudgetExceededError, share_budget
from atomics.eval.codegen.runner import run_codegen
from atomics.eval.codereview.runner import run_codereview
from atomics.eval.multiturn.runner import run_multiturn
from atomics.eval.rag.runner import run_rag
from atomics.eval.redblue.runner import run_redblue
from atomics.eval.refusal.runner import run_refusal
from atomics.eval.runner import run_eval
from atomics.eval.toolcall.fixtures import ALL_FIXTURES as TOOLCALL_FIXTURES
from atomics.eval.toolcall.runner import run_toolcall_suite
from atomics.models import BurnTier
from atomics.providers.base import BaseProvider
from atomics.providers.factory import ProviderConfigError, make_provider

SUPPORTED_EVAL_SUITES = frozenset(
    {
        "accuracy",
        "rag",
        "multiturn",
        "adversarial",
        "codegen",
        "refusal",
        "redblue",
        "toolcall",
        "codereview",
    }
)


def validate_eval_suite(suite: str) -> str:
    """Normalize and validate an eval suite name."""
    normalized = suite.lower()
    if normalized not in SUPPORTED_EVAL_SUITES:
        raise ValueError(f"Unsupported eval suite: {normalized}")
    return normalized


def _provider_for(name: str, model: str | None, host: str | None = None) -> BaseProvider:
    settings = load_settings()
    try:
        if name == "vllm":
            return make_provider(name, model, None, settings, vllm_host=host)
        return make_provider(name, model, host, settings)
    except ProviderConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _guarded_providers(
    payload: EvalRequest,
) -> tuple[BaseProvider, BaseProvider]:
    """Build the model and judge for an eval request, sharing one budget.

    Always metered, unlike the CLI. A local operator spends their own money
    deliberately; an API caller is remote and authenticated by a shared key, so
    the ceiling is the only thing between them and the account limit.
    """
    provider = _provider_for(payload.provider, payload.model, payload.host)
    judge_host = payload.host if payload.provider == "ollama" else None
    judge_provider = _provider_for("ollama", payload.judge_model, judge_host)
    budget = EvalBudget(budget_limit_usd=payload.budget_usd)
    guarded = share_budget(budget, provider, judge_provider)
    return guarded[0], guarded[1]


def _can_row(fr: Any) -> bool:
    if isinstance(fr, (int, float, str, bool)) or fr is None:
        return False
    if isinstance(fr, dict):
        return bool(fr.get("id") or fr.get("fixture"))
    return hasattr(fr, "fixture") or bool(getattr(fr, "id", None))


def _fixture_rows(results: Any) -> list[FixtureRow]:
    rows: list[FixtureRow] = []
    for fr in results or []:
        if not _can_row(fr):
            continue
        try:
            row = fixture_row(fr)
        except (AttributeError, TypeError, KeyError, ValueError):
            continue
        if row["id"]:
            rows.append(row)
    return rows


def _summary_fixture_items(summary: Any) -> Any:
    for attr in ("fixture_results", "conversation_results", "fixtures", "results"):
        items = getattr(summary, attr, None)
        if items:
            return items
    return []


def _completed_fixtures(summary: Any, job: Job | None) -> list[FixtureRow]:
    if job is not None and isinstance(job.result, dict):
        live = job.result.get("fixtures")
        if isinstance(live, list) and live:
            return list(live)
    return _fixture_rows(_summary_fixture_items(summary))


def _eval_reporter(payload: EvalRequest, job: Job | None, suite: str) -> EvalJobReporter | None:
    if job is None:
        return None
    request = job.request or {}
    host = request.get("host")
    return EvalJobReporter(
        job,
        suite=suite,
        provider=payload.provider,
        model=payload.model or request.get("model"),
        judge_model=payload.judge_model or request.get("judge_model"),
        host=host if isinstance(host, str) else None,
        total=eval_fixture_total(payload),
    )


def _summary_totals(summary: Any) -> tuple[int, float]:
    """Extract token/cost totals across suite summary shapes."""
    tokens = getattr(summary, "total_tokens", None)
    cost = getattr(summary, "total_cost_usd", None)
    if tokens is not None and cost is not None:
        return int(tokens), float(cost)

    fixture_results = getattr(summary, "fixture_results", None) or []
    if cost is None:
        cost = sum(float(getattr(fr, "estimated_cost_usd", 0.0)) for fr in fixture_results)
    if tokens is None:
        tokens = 0
        for fr in fixture_results:
            fr_tokens = getattr(fr, "total_tokens", None)
            if fr_tokens is not None:
                tokens += int(fr_tokens)
                continue
            attempts = getattr(fr, "attempts", None) or []
            tokens += sum(int(getattr(a, "total_tokens", 0) or 0) for a in attempts)
    return int(tokens or 0), float(cost or 0.0)


def _overall_score(summary: Any) -> float | None:
    for attr in (
        "overall_score",
        "overall_rag_score",
        "overall_pass_rate",
        "overall_resilience",
        "avg_conversation_score",
        "calibration_score",
        "overall_quality",
        "review_score",
        "dangerous_call_rate",
    ):
        value = getattr(summary, attr, None)
        if value is not None:
            return float(value)
    return None


async def run_benchmark_from_request(
    payload: RunRequest, job: Job | None = None
) -> dict[str, Any]:
    from atomics.benchmark.tiers import get_tier_profile
    from atomics.core.engine import LoopEngine
    from atomics.storage.repository import MetricsRepository

    settings = load_settings()
    try:
        provider = _provider_for(payload.provider, payload.model, payload.host)
        tier = BurnTier(payload.tier)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    profile = get_tier_profile(tier)
    repo = MetricsRepository(settings.db_path)
    reporter = None
    request: dict[str, Any] = {}
    if job is not None:
        request = job.request or {}
        reporter = LoadJobReporter(
            job,
            kind="run",
            meta={
                "provider": payload.provider,
                "model": payload.model or request.get("model"),
                "tier": payload.tier,
                "host": request.get("host"),
            },
            rows_key="task_rows",
            total=int(payload.iterations),
        )
    try:
        engine = LoopEngine(
            provider=provider,
            repo=repo,
            settings=settings,
            tier=tier,
            interval_override=payload.interval,
            model_override=payload.model,
            trigger="api",
            thinking=payload.thinking,
            effort=payload.effort,
            reasoning_mode=payload.reasoning_mode,
            on_task_start=(
                None
                if reporter is None
                else lambda name: reporter.start(
                    {"task": name, "model": payload.model or request.get("model")}
                )
            ),
            on_task_done=(
                None if reporter is None else lambda result: reporter.done(burn_row(result))
            ),
        )
        summary = await engine.run(max_iterations=payload.iterations)
        if summary is None:
            raise RuntimeError("Benchmark run produced no summary")
        body: dict[str, Any] = {
            "run_id": summary.run_id,
            "tasks": summary.total_tasks,
            "success": summary.successful_tasks,
            "failed": summary.failed_tasks,
            "total_tokens": summary.total_tokens,
            "total_cost_usd": summary.total_cost_usd,
            "provider": payload.provider,
            "model": payload.model or profile.preferred_model or settings.default_model,
            "tier": payload.tier,
        }
        if reporter is not None and job is not None and isinstance(job.result, dict):
            body["task_rows"] = list(job.result.get("task_rows") or [])
        return body
    except HTTPException:
        raise
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        repo.close()


async def run_eval_from_request(
    payload: EvalRequest, job: Job | None = None
) -> dict[str, Any]:
    """Run the accuracy eval suite for an API request."""
    try:
        provider, judge_provider = _guarded_providers(payload)
        fixtures = select_eval_fixtures(payload.fixtures)
        reporter = None
        if job is not None:
            request = job.request or {}
            reporter = EvalJobReporter(
                job,
                suite="accuracy",
                provider=payload.provider,
                model=payload.model or request.get("model"),
                judge_model=payload.judge_model or request.get("judge_model"),
                host=request.get("host"),
                total=eval_fixture_total(payload),
            )
        summary = await run_eval(
            provider,
            judge_provider=judge_provider,
            model=payload.model,
            judge_model=payload.judge_model,
            run_id=None,
            thinking=payload.thinking,
            effort=payload.effort,
            reasoning_mode=payload.reasoning_mode,
            fixtures=fixtures,
            on_phase=reporter.phase if reporter is not None else None,
            on_fixture_done=reporter.fixture_done if reporter is not None else None,
        )
    except HTTPException:
        raise
    # Surfaces on the job as EvalBudgetExceededError rather than being flattened
    # into a 400. The request was valid; the run stopped on a ceiling, and the
    # caller needs to tell those apart.
    except EvalBudgetExceededError:
        raise
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    host = (job.request or {}).get("host") if job is not None else None
    return {
        "provider": payload.provider,
        "model": payload.model,
        "judge_model": payload.judge_model,
        "host": host,
        "overall_accuracy": summary.overall_accuracy,
        "fixtures_run": len(summary.fixture_results),
        "total_tokens": summary.total_tokens,
        "total_cost_usd": summary.total_cost_usd,
        "fixtures": _fixture_rows(summary.fixture_results),
    }


async def run_eval_suite(payload: EvalRequest, job: Job | None = None) -> dict[str, Any]:
    """Dispatch eval request to the correct runner and normalize the response."""
    suite = validate_eval_suite(payload.suite)

    if suite == "accuracy":
        result = await run_eval_from_request(payload, job=job)
        return {"suite": suite, **result}

    reporter = _eval_reporter(payload, job, suite)
    on_done = reporter.fixture_done if reporter is not None else None
    on_phase = reporter.phase if reporter is not None else None

    def on_toolcall_done(_index: int, _fixture: object, aggregated: object) -> None:
        if reporter is not None:
            reporter.fixture_done(aggregated)

    provider, judge_provider = _guarded_providers(payload)

    summary: Any
    try:
        if suite == "rag":
            summary = await run_rag(
                provider,
                judge_provider=judge_provider,
                model=payload.model,
                judge_model=payload.judge_model,
                thinking=payload.thinking,
                effort=payload.effort,
                reasoning_mode=payload.reasoning_mode,
                on_fixture_done=on_done,
                on_phase=on_phase,
            )
            fixtures_run = len(summary.fixture_results)
        elif suite == "multiturn":
            summary = await run_multiturn(
                provider,
                judge_provider=judge_provider,
                model=payload.model,
                judge_model=payload.judge_model,
                thinking=payload.thinking,
                effort=payload.effort,
                reasoning_mode=payload.reasoning_mode,
                on_conversation_done=on_done,
                on_phase=on_phase,
            )
            fixtures_run = len(summary.conversation_results)
        elif suite == "adversarial":
            summary = await run_adversarial(
                provider,
                judge_provider=judge_provider,
                model=payload.model,
                judge_model=payload.judge_model,
                thinking=payload.thinking,
                effort=payload.effort,
                reasoning_mode=payload.reasoning_mode,
                on_fixture_done=on_done,
                on_phase=on_phase,
            )
            fixtures_run = len(summary.fixture_results)
        elif suite == "codegen":
            summary = await run_codegen(
                provider,
                model=payload.model,
                thinking=payload.thinking,
                effort=payload.effort,
                reasoning_mode=payload.reasoning_mode,
                on_fixture_done=on_done,
                on_phase=on_phase,
            )
            fixtures_run = len(summary.fixture_results)
        elif suite == "refusal":
            summary = await run_refusal(
                provider,
                judge_provider=judge_provider,
                model=payload.model,
                judge_model=payload.judge_model,
                thinking=payload.thinking,
                effort=payload.effort,
                reasoning_mode=payload.reasoning_mode,
                on_fixture_done=on_done,
                on_phase=on_phase,
            )
            fixtures_run = len(summary.fixture_results)
        elif suite == "redblue":
            summary = await run_redblue(
                provider,
                judge_provider=judge_provider,
                model=payload.model,
                judge_model=payload.judge_model,
                thinking=payload.thinking,
                effort=payload.effort,
                reasoning_mode=payload.reasoning_mode,
                on_fixture_done=on_done,
                on_phase=on_phase,
            )
            fixtures_run = len(summary.fixture_results)
        elif suite == "codereview":
            summary = await run_codereview(
                provider,
                judge_provider=judge_provider,
                model=payload.model,
                judge_model=payload.judge_model,
                thinking=payload.thinking,
                effort=payload.effort,
                reasoning_mode=payload.reasoning_mode,
                on_fixture_done=on_done,
                on_phase=on_phase,
            )
            fixtures_run = len(summary.fixture_results)
        elif suite == "toolcall":
            # Keyword-only runner; fixtures default to the full catalog so an
            # API caller gets the same suite the CLI does without a prompt list.
            summary = await run_toolcall_suite(
                provider=provider,
                model=payload.model,
                judge_provider=judge_provider,
                fixtures=TOOLCALL_FIXTURES,
                judge_model=payload.judge_model,
                thinking=payload.thinking,
                effort=payload.effort,
                reasoning_mode=payload.reasoning_mode,
                on_fixture_done=on_toolcall_done if reporter is not None else None,
                on_phase=on_phase,
            )
            fixtures_run = len(summary.fixtures)
        else:  # pragma: no cover - guarded by validate_eval_suite
            raise ValueError(f"Unsupported eval suite: {suite}")
    except HTTPException:
        raise
    except EvalBudgetExceededError:
        raise
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    total_tokens, total_cost_usd = _summary_totals(summary)
    host = (job.request or {}).get("host") if job is not None else None
    return {
        "suite": suite,
        "provider": payload.provider,
        "model": payload.model,
        "host": host,
        "overall_score": _overall_score(summary),
        "fixtures_run": fixtures_run,
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost_usd,
        "fixtures": _completed_fixtures(summary, job),
    }
