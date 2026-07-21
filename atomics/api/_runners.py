"""Async runners used by the API server routes."""

from __future__ import annotations

from typing import Any

import click
from fastapi import HTTPException

from atomics.api.models import EvalRequest, RunRequest
from atomics.commands.common import _make_provider
from atomics.config import load_settings
from atomics.eval.adversarial.runner import run_adversarial
from atomics.eval.codegen.runner import run_codegen
from atomics.eval.multiturn.runner import run_multiturn
from atomics.eval.rag.runner import run_rag
from atomics.eval.runner import run_eval
from atomics.models import BurnTier
from atomics.providers.base import BaseProvider

SUPPORTED_EVAL_SUITES = frozenset(
    {"accuracy", "rag", "multiturn", "adversarial", "codegen"}
)


def validate_eval_suite(suite: str) -> str:
    """Normalize and validate an eval suite name."""
    normalized = suite.lower()
    if normalized not in SUPPORTED_EVAL_SUITES:
        raise ValueError(f"Unsupported eval suite: {normalized}")
    return normalized


def _provider_for(name: str, model: str | None) -> BaseProvider:
    settings = load_settings()
    try:
        return _make_provider(name, model, None, settings)
    except click.ClickException as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
    ):
        value = getattr(summary, attr, None)
        if value is not None:
            return float(value)
    return None


async def run_benchmark_from_request(payload: RunRequest) -> dict[str, Any]:
    from atomics.core.engine import LoopEngine
    from atomics.storage.repository import MetricsRepository
    from atomics.tiers import get_tier_profile

    settings = load_settings()
    try:
        provider = _provider_for(payload.provider, payload.model)
        tier = BurnTier(payload.tier)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    profile = get_tier_profile(tier)
    repo = MetricsRepository(settings.db_path)
    try:
        engine = LoopEngine(
            provider=provider,
            repo=repo,
            settings=settings,
            tier=tier,
            interval_override=payload.interval,
            model_override=payload.model,
            trigger="api",
        )
        summary = await engine.run(max_iterations=payload.iterations)
        if summary is None:
            raise RuntimeError("Benchmark run produced no summary")
        return {
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
    except HTTPException:
        raise
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        repo.close()


async def run_eval_from_request(payload: EvalRequest) -> dict[str, Any]:
    """Run the accuracy eval suite for an API request."""
    try:
        provider = _provider_for(payload.provider, payload.model)
        judge_provider = _provider_for("ollama", payload.judge_model)
        summary = await run_eval(
            provider,
            judge_provider=judge_provider,
            model=payload.model,
            judge_model=payload.judge_model,
            run_id=None,
        )
    except HTTPException:
        raise
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "provider": payload.provider,
        "model": payload.model,
        "judge_model": payload.judge_model,
        "overall_accuracy": summary.overall_accuracy,
        "fixtures_run": len(summary.fixture_results),
        "total_tokens": summary.total_tokens,
        "total_cost_usd": summary.total_cost_usd,
    }


async def run_eval_suite(payload: EvalRequest) -> dict[str, Any]:
    """Dispatch eval request to the correct runner and normalize the response."""
    suite = validate_eval_suite(payload.suite)

    if suite == "accuracy":
        result = await run_eval_from_request(payload)
        return {"suite": suite, **result}

    provider = _provider_for(payload.provider, payload.model)
    judge_provider = _provider_for("ollama", payload.judge_model)

    summary: Any
    try:
        if suite == "rag":
            summary = await run_rag(
                provider,
                judge_provider=judge_provider,
                model=payload.model,
                judge_model=payload.judge_model,
            )
            fixtures_run = len(summary.fixture_results)
        elif suite == "multiturn":
            summary = await run_multiturn(
                provider,
                judge_provider=judge_provider,
                model=payload.model,
                judge_model=payload.judge_model,
            )
            fixtures_run = len(summary.conversation_results)
        elif suite == "adversarial":
            summary = await run_adversarial(
                provider,
                judge_provider=judge_provider,
                model=payload.model,
                judge_model=payload.judge_model,
            )
            fixtures_run = len(summary.fixture_results)
        elif suite == "codegen":
            summary = await run_codegen(
                provider,
                model=payload.model,
            )
            fixtures_run = len(summary.fixture_results)
        else:  # pragma: no cover - guarded by validate_eval_suite
            raise ValueError(f"Unsupported eval suite: {suite}")
    except HTTPException:
        raise
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    total_tokens, total_cost_usd = _summary_totals(summary)
    return {
        "suite": suite,
        "provider": payload.provider,
        "model": payload.model,
        "overall_score": _overall_score(summary),
        "fixtures_run": fixtures_run,
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost_usd,
    }
