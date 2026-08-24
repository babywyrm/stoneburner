"""Resolved job request and live eval progress helpers.

The job document is the operator view. These helpers never talk HTTP; routes
and runners call them while mutating an in-memory `Job`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypedDict

from atomics.api.jobs import Job
from atomics.api.models import EvalRequest, SoakRequest, StressRequest, SweepRequest
from atomics.config import AtomicsSettings
from atomics.eval.adversarial import ALL_FIXTURES as ADVERSARIAL_FIXTURES
from atomics.eval.codegen.fixtures import ALL_CODEGEN_FIXTURES
from atomics.eval.codereview.fixtures import SECURE_CODE_FIXTURES
from atomics.eval.fixtures import EVAL_FIXTURES, EvalFixture
from atomics.eval.multiturn.fixtures import ALL_MULTITURN_FIXTURES
from atomics.eval.rag.fixtures import ALL_RAG_FIXTURES
from atomics.eval.redblue.fixtures import ALL_FIXTURES as REDBLUE_FIXTURES
from atomics.eval.refusal.fixtures import REFUSAL_FIXTURES
from atomics.eval.toolcall.fixtures import ALL_FIXTURES as TOOLCALL_FIXTURES
from atomics.load.stress import stress_concurrency_levels

RESPONSE_LIMIT = 500

SuiteCatalog = Sequence[object]

_SUITE_CATALOGS: dict[str, SuiteCatalog] = {
    "rag": ALL_RAG_FIXTURES,
    "multiturn": ALL_MULTITURN_FIXTURES,
    "adversarial": ADVERSARIAL_FIXTURES,
    "codegen": ALL_CODEGEN_FIXTURES,
    "refusal": REFUSAL_FIXTURES,
    "redblue": REDBLUE_FIXTURES,
    "toolcall": TOOLCALL_FIXTURES,
    "codereview": SECURE_CODE_FIXTURES,
}


class SweepJobRow(TypedDict):
    model: str
    suite: str
    ok: bool
    headline: float | None
    error: str | None
    tool_capable: bool | None
    exit_code: int


class FixtureRow(TypedDict):
    id: str
    status: Literal["success", "failed"]
    score: float | None
    tokens: int
    latency_ms: float
    response: str | None
    error: str | None


def truncate_response(text: str | None) -> str | None:
    if text is None:
        return None
    if len(text) <= RESPONSE_LIMIT:
        return text
    return text[:RESPONSE_LIMIT]


def resolve_inference_host(
    provider: str, host: str | None, settings: AtomicsSettings
) -> str | None:
    if host:
        return host
    if provider == "ollama":
        return settings.ollama_host
    if provider == "vllm":
        return settings.vllm_host
    if provider == "llamacpp":
        return settings.llamacpp_host
    return None


def resolve_eval_request(payload: EvalRequest, settings: AtomicsSettings) -> dict[str, Any]:
    request: dict[str, Any] = {
        "suite": payload.suite,
        "provider": payload.provider,
        "model": payload.model or settings.ollama_model,
        "judge_model": payload.judge_model or settings.ollama_model,
        "host": resolve_inference_host(payload.provider, payload.host, settings),
    }
    if payload.effort is not None:
        request["effort"] = payload.effort
    if payload.reasoning_mode is not None:
        request["reasoning_mode"] = payload.reasoning_mode
    if payload.thinking is not None:
        request["thinking"] = payload.thinking
    request["budget_usd"] = payload.budget_usd
    return request


def short_request(request: dict[str, Any] | None) -> dict[str, Any] | None:
    if not request:
        return None
    out: dict[str, Any] = {
        key: request[key]
        for key in ("suite", "model", "host", "models", "suites")
        if key in request
    }
    return out or None


def select_eval_fixtures(ids: list[str] | None) -> list[EvalFixture] | None:
    if ids is None:
        return None
    by_id = {fixture.id: fixture for fixture in EVAL_FIXTURES}
    return [by_id[item] for item in ids if item in by_id]


def eval_fixture_total(payload: EvalRequest) -> int:
    if payload.suite == "accuracy":
        selected = select_eval_fixtures(payload.fixtures)
        return len(EVAL_FIXTURES) if selected is None else len(selected)
    catalog = _SUITE_CATALOGS.get(payload.suite)
    return len(catalog) if catalog is not None else 0


def initial_eval_progress(payload: EvalRequest) -> dict[str, Any]:
    return {
        "current": 0,
        "total": eval_fixture_total(payload),
        "in_flight": None,
        "trail": [],
    }


def _append_trail(progress: dict[str, Any], entry: dict[str, Any], *, cap: int) -> None:
    trail = list(progress.get("trail") or [])
    if cap <= 0 or len(trail) < cap:
        trail.append(entry)
    progress["trail"] = trail


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_float(value: Any) -> float | None:
    if value is None or callable(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_id(fr: Any) -> str:
    fixture = _field(fr, "fixture")
    if fixture is not None:
        ident = getattr(fixture, "id", None)
        if ident:
            return str(ident)
    ident = _field(fr, "id")
    return str(ident) if ident else ""


def _row_failed(fr: Any) -> bool:
    task = _field(fr, "task_result")
    if task is not None:
        status = getattr(task, "status", None)
        if getattr(status, "value", status) == "failed":
            return True
    return bool(_field(fr, "error"))


def _row_error(fr: Any) -> str | None:
    task = _field(fr, "task_result")
    if task is not None:
        message = getattr(task, "error_message", None)
        if message:
            return str(message)
    error = _field(fr, "error")
    return str(error) if error else None


def _attempt_judge_score(attempts: Any) -> float | None:
    if not attempts:
        return None
    first = attempts[0]
    judge = _field(first, "judge")
    if judge is None or getattr(judge, "parse_failed", False):
        return None
    return _as_float(getattr(judge, "score", None))


def _row_score(fr: Any, *, failed: bool) -> float | None:
    if failed:
        return None
    judge = _field(fr, "judge")
    if judge is not None and not getattr(judge, "parse_failed", False):
        scored = _as_float(getattr(judge, "score", None))
        if scored is not None:
            return scored
    for name in ("score", "overall_score", "pass_rate"):
        scored = _as_float(_field(fr, name))
        if scored is not None:
            return scored
    resistance = _field(fr, "resistance")
    if resistance is not None:
        scored = _as_float(getattr(resistance, "score", None))
        if scored is not None:
            return scored
    scored = _attempt_judge_score(_field(fr, "attempts"))
    if scored is not None:
        return scored
    outcome = _field(fr, "tool_outcome")
    if outcome is None:
        return None
    key = str(getattr(outcome, "value", outcome))
    if key == "dangerous_call":
        return 0.0
    if key == "safe_call":
        return 1.0
    return None


def _row_tokens(fr: Any) -> int:
    task = _field(fr, "task_result")
    if task is not None:
        return int(getattr(task, "total_tokens", 0) or 0)
    direct = _field(fr, "total_tokens")
    if direct is not None:
        return int(direct)
    attempts = _field(fr, "attempts") or []
    if attempts:
        return sum(int(_field(attempt, "total_tokens") or 0) for attempt in attempts)
    runs = _field(fr, "runs") or []
    if runs:
        return sum(int(_field(run, "total_tokens") or 0) for run in runs)
    return 0


def _row_latency(fr: Any) -> float:
    task = _field(fr, "task_result")
    if task is not None:
        return round(float(getattr(task, "latency_ms", 0.0) or 0.0), 1)
    return round(float(_field(fr, "latency_ms") or 0.0), 1)


def _row_response(fr: Any, *, failed: bool) -> str | None:
    if failed:
        return None
    task = _field(fr, "task_result")
    if task is not None:
        return truncate_response(getattr(task, "response", None))
    for name in ("response_text", "review_text", "response", "tool_text", "prose_text"):
        value = _field(fr, name)
        if value:
            return truncate_response(str(value))
    return None


def row_cost(fr: Any) -> float:
    task = _field(fr, "task_result")
    if task is not None:
        return float(getattr(task, "estimated_cost_usd", 0.0) or 0.0)
    for name in ("estimated_cost_usd", "cost_usd"):
        value = _field(fr, name)
        if value is not None:
            return float(value)
    return 0.0


def fixture_row(fr: Any) -> FixtureRow:
    failed = _row_failed(fr)
    return {
        "id": _row_id(fr),
        "status": "failed" if failed else "success",
        "score": _row_score(fr, failed=failed),
        "tokens": _row_tokens(fr),
        "latency_ms": _row_latency(fr),
        "response": _row_response(fr, failed=failed),
        "error": _row_error(fr),
    }


def sweep_job_total(payload: SweepRequest) -> int:
    return len(payload.models) * len(payload.suites)


def initial_sweep_progress(payload: SweepRequest) -> dict[str, Any]:
    return {"current": 0, "total": sweep_job_total(payload), "in_flight": None, "trail": []}


def stress_job_total(payload: StressRequest) -> int:
    return len(stress_concurrency_levels(payload.max_concurrency))


def soak_job_total(payload: SoakRequest) -> int:
    """How many sampler windows finish before duration cancels the last sleep.

    The sampler sleeps `sample_interval`, records, repeats. Stop fires at
    `duration_seconds`, so a tick at t == duration is cancelled. 30s / 10s
    yields samples at 10s and 20s — two rows, not three.
    """
    duration = int(payload.duration_seconds)
    interval = int(payload.sample_interval)
    if interval <= 0:
        return 0
    return max(0, (duration - 1) // interval)


def initial_stress_progress(payload: StressRequest) -> dict[str, Any]:
    return {"current": 0, "total": stress_job_total(payload), "in_flight": None, "trail": []}


def initial_soak_progress(payload: SoakRequest) -> dict[str, Any]:
    return {"current": 0, "total": soak_job_total(payload), "in_flight": None, "trail": []}


def sweep_row(result: Any) -> SweepJobRow:
    headline = getattr(result, "headline", None)
    scored: float | None
    try:
        scored = None if headline is None else float(headline)
    except (TypeError, ValueError):
        scored = None
    capable = getattr(result, "tool_capable", None)
    return {
        "model": str(getattr(result, "model", "") or ""),
        "suite": str(getattr(result, "suite", "") or ""),
        "ok": bool(getattr(result, "ok", False)),
        "headline": scored,
        "error": getattr(result, "error", None),
        "tool_capable": None if capable is None else bool(capable),
        "exit_code": int(getattr(result, "exit_code", 0) or 0),
    }


def payload_request(payload: Any, settings: AtomicsSettings) -> dict[str, Any]:
    """Echo a non-eval submit payload plus resolved host."""
    data = payload.model_dump(exclude_none=True)
    provider = getattr(payload, "provider", None)
    if provider:
        host = resolve_inference_host(
            str(provider), getattr(payload, "host", None), settings
        )
        if host:
            data["host"] = host
    return data


class EvalJobReporter:
    """Mutate a job as accuracy fixtures start and finish."""

    def __init__(
        self,
        job: Job,
        *,
        suite: str,
        provider: str,
        model: str | None,
        judge_model: str | None,
        host: str | None,
        total: int,
    ) -> None:
        self.job = job
        self._meta = {
            "suite": suite,
            "provider": provider,
            "model": model,
            "judge_model": judge_model,
            "host": host,
        }
        job.progress = {"current": 0, "total": total, "in_flight": None, "trail": []}

    def phase(self, fixture_id: str, phase: str, model: str | None) -> None:
        progress = dict(self.job.progress or {})
        entry = {
            "fixture_id": fixture_id,
            "phase": phase,
            "model": model,
        }
        progress["in_flight"] = entry
        _append_trail(progress, entry, cap=2 * int(progress.get("total") or 0))
        self.job.progress = progress

    def fixture_done(self, fr: Any) -> None:
        row = fixture_row(fr)
        result = self.job.result
        if result is None:
            result = {
                **self._meta,
                "overall_accuracy": None,
                "overall_score": None,
                "fixtures_run": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "fixtures": [],
            }
            self.job.result = result
        result["fixtures"].append(row)
        result["fixtures_run"] = len(result["fixtures"])
        result["total_tokens"] = int(result["total_tokens"]) + int(row["tokens"])
        result["total_cost_usd"] = round(float(result["total_cost_usd"]) + row_cost(fr), 6)
        prior = self.job.progress or {}
        self.job.progress = {
            "current": result["fixtures_run"],
            "total": prior.get("total"),
            "in_flight": None,
            "trail": list(prior.get("trail") or []),
        }


class SweepJobReporter:
    """Mutate a job as each models×suites cell starts and finishes."""

    def __init__(
        self,
        job: Job,
        *,
        provider: str,
        models: list[str],
        suites: list[str],
        runs: int,
        budget_usd: float,
        total: int,
    ) -> None:
        self.job = job
        self._meta = {
            "provider": provider,
            "models": list(models),
            "suites": list(suites),
            "runs": runs,
            "budget_usd": budget_usd,
        }
        job.progress = {"current": 0, "total": total, "in_flight": None, "trail": []}

    def start(self, model: str, suite: str) -> None:
        progress = dict(self.job.progress or {})
        entry = {"model": model, "suite": suite}
        progress["in_flight"] = entry
        _append_trail(progress, entry, cap=int(progress.get("total") or 0))
        self.job.progress = progress

    def done(self, result: Any) -> None:
        row = sweep_row(result)
        body = self.job.result
        if body is None:
            body = {**self._meta, "ok": 0, "fail": 0, "jobs": []}
            self.job.result = body
        body["jobs"].append(row)
        body["ok"] = sum(1 for item in body["jobs"] if item.get("ok"))
        body["fail"] = sum(1 for item in body["jobs"] if not item.get("ok"))
        prior = self.job.progress or {}
        self.job.progress = {
            "current": len(body["jobs"]),
            "total": prior.get("total"),
            "in_flight": None,
            "trail": list(prior.get("trail") or []),
        }


class LoadJobReporter:
    """Mutate a job as each stress phase or soak sample starts and finishes."""

    def __init__(
        self,
        job: Job,
        *,
        kind: str,
        meta: dict[str, Any],
        rows_key: str,
        total: int,
    ) -> None:
        self.job = job
        self._kind = kind
        self._meta = dict(meta)
        self._rows_key = rows_key
        job.progress = {"current": 0, "total": total, "in_flight": None, "trail": []}

    def start(self, in_flight: dict[str, Any]) -> None:
        progress = dict(self.job.progress or {})
        entry = dict(in_flight)
        progress["in_flight"] = entry
        _append_trail(progress, entry, cap=int(progress.get("total") or 0))
        self.job.progress = progress

    def done(self, row: dict[str, Any]) -> None:
        body = self.job.result
        if body is None:
            body = {**self._meta, self._rows_key: []}
            self.job.result = body
        body[self._rows_key].append(row)
        prior = self.job.progress or {}
        self.job.progress = {
            "current": len(body[self._rows_key]),
            "total": prior.get("total"),
            "in_flight": None,
            "trail": list(prior.get("trail") or []),
        }
