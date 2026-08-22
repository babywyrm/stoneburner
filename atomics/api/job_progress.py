"""Resolved job request and live eval progress helpers.

The job document is the operator view. These helpers never talk HTTP; routes
and runners call them while mutating an in-memory `Job`.
"""

from __future__ import annotations

from typing import Any

from atomics.api.jobs import Job
from atomics.api.models import EvalRequest
from atomics.config import AtomicsSettings
from atomics.eval.fixtures import EVAL_FIXTURES, EvalFixture

RESPONSE_LIMIT = 500


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
    out = {key: request[key] for key in ("suite", "model", "host") if key in request}
    return out or None


def select_eval_fixtures(ids: list[str] | None) -> list[EvalFixture] | None:
    if ids is None:
        return None
    by_id = {fixture.id: fixture for fixture in EVAL_FIXTURES}
    return [by_id[item] for item in ids if item in by_id]


def eval_fixture_total(payload: EvalRequest) -> int:
    selected = select_eval_fixtures(payload.fixtures)
    return len(EVAL_FIXTURES) if selected is None else len(selected)


def initial_eval_progress(payload: EvalRequest) -> dict[str, Any]:
    return {"current": 0, "total": eval_fixture_total(payload), "in_flight": None}


def fixture_row(fr: Any) -> dict[str, Any]:
    task = fr.task_result
    failed = getattr(task.status, "value", task.status) == "failed"
    judge = fr.judge
    score = None
    if judge is not None and not getattr(judge, "parse_failed", False):
        score = judge.score
    return {
        "id": fr.fixture.id,
        "status": "failed" if failed else "success",
        "score": None if failed else score,
        "tokens": int(getattr(task, "total_tokens", 0) or 0),
        "latency_ms": round(float(getattr(task, "latency_ms", 0.0) or 0.0), 1),
        "response": None if failed else truncate_response(getattr(task, "response", None)),
        "error": getattr(task, "error_message", None) or None,
    }


def payload_request(payload: Any, settings: AtomicsSettings) -> dict[str, Any]:
    """Echo a non-eval submit payload plus resolved host."""
    data = payload.model_dump(exclude_none=True)
    provider = getattr(payload, "provider", None)
    if provider:
        host = resolve_inference_host(str(provider), None, settings)
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
        job.progress = {"current": 0, "total": total, "in_flight": None}

    def phase(self, fixture_id: str, phase: str, model: str | None) -> None:
        progress = dict(self.job.progress or {})
        progress["in_flight"] = {
            "fixture_id": fixture_id,
            "phase": phase,
            "model": model,
        }
        self.job.progress = progress

    def fixture_done(self, fr: Any) -> None:
        row = fixture_row(fr)
        result = self.job.result
        if result is None:
            result = {
                **self._meta,
                "overall_accuracy": None,
                "fixtures_run": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "fixtures": [],
            }
            self.job.result = result
        result["fixtures"].append(row)
        result["fixtures_run"] = len(result["fixtures"])
        result["total_tokens"] = int(result["total_tokens"]) + int(row["tokens"])
        cost = float(getattr(fr.task_result, "estimated_cost_usd", 0.0) or 0.0)
        result["total_cost_usd"] = round(float(result["total_cost_usd"]) + cost, 6)
        self.job.progress = {
            "current": result["fixtures_run"],
            "total": (self.job.progress or {}).get("total"),
            "in_flight": None,
        }
