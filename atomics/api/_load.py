"""Bounded stress and soak jobs for the API.

The CLI can ramp to high concurrency or soak for hours. A remote caller gets
one named model, a required budget, and hard caps on concurrency and duration.
No contention mode, no profile YAML, no baselines.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from atomics.api._runners import _provider_for
from atomics.api.job_progress import (
    LoadJobReporter,
    soak_job_total,
    stress_job_total,
)
from atomics.api.jobs import Job
from atomics.api.models import MAX_LOAD_PREDICT, SoakRequest, StressRequest
from atomics.eval.budget import BudgetMeter, EvalBudget, EvalBudgetExceededError
from atomics.load.soak import SoakResult, SoakSample, run_soak_provider
from atomics.load.stress import ConcurrencyResult, StressResult, run_stress_provider
from atomics.providers.base import BaseProvider


def _metered(
    provider_name: str, model: str, budget_usd: float, host: str | None = None
) -> BaseProvider:
    meter = BudgetMeter(EvalBudget(budget_limit_usd=budget_usd))
    return meter.wrap(_provider_for(provider_name, model, host))


def _phase_row(phase: ConcurrencyResult) -> dict[str, Any]:
    return {
        "concurrency": phase.concurrency,
        "requests": phase.requests,
        "failed": phase.failed,
        "aggregate_tps": phase.aggregate_tps,
        "avg_latency_ms": phase.avg_latency_ms,
        "p95_latency_ms": phase.p95_latency_ms,
    }


def _sample_row(sample: SoakSample) -> dict[str, Any]:
    return {
        "elapsed_seconds": sample.elapsed_seconds,
        "requests": sample.requests,
        "failed": sample.failed,
        "aggregate_tps": sample.aggregate_tps,
        "avg_latency_ms": sample.avg_latency_ms,
        "p95_latency_ms": sample.p95_latency_ms,
    }


def _stress_payload(payload: StressRequest, result: StressResult) -> dict[str, Any]:
    return {
        "provider": payload.provider,
        "model": result.model,
        "budget_usd": payload.budget_usd,
        "max_concurrency": payload.max_concurrency,
        "phase_seconds": payload.phase_seconds,
        "duration_seconds": result.duration_seconds,
        "total_tokens": result.total_tokens,
        "total_requests": result.total_requests,
        "total_failed": result.total_failed,
        "peak_tps": result.peak_tps,
        "saturation_concurrency": result.saturation_concurrency,
        "total_cost_usd": result.total_cost_usd,
        "phases": [_phase_row(phase) for phase in result.phases],
    }


def _soak_payload(payload: SoakRequest, result: SoakResult) -> dict[str, Any]:
    return {
        "provider": payload.provider,
        "model": result.model,
        "budget_usd": payload.budget_usd,
        "concurrency": result.concurrency,
        "duration_seconds": result.duration_seconds,
        "sample_interval": payload.sample_interval,
        "actual_duration_seconds": result.actual_duration_seconds,
        "total_requests": result.total_requests,
        "total_failed": result.total_failed,
        "total_tokens": result.total_tokens,
        "avg_tps": result.avg_tps,
        "peak_tps": result.peak_tps,
        "min_tps": result.min_tps,
        "avg_p95_ms": result.avg_p95_ms,
        "error_rate": result.error_rate,
        "throughput_drift_pct": result.throughput_drift_pct,
        "latency_drift_pct": result.latency_drift_pct,
        "verdict": result.verdict,
        "total_cost_usd": result.total_cost_usd,
        "samples": [_sample_row(sample) for sample in result.samples],
    }


def _stress_hooks(
    payload: StressRequest, job: Job | None
) -> dict[str, Any]:
    if job is None:
        return {}
    reporter = LoadJobReporter(
        job,
        kind="stress",
        meta={
            "provider": payload.provider,
            "model": payload.model,
            "budget_usd": payload.budget_usd,
            "max_concurrency": payload.max_concurrency,
            "phase_seconds": payload.phase_seconds,
        },
        rows_key="phases",
        total=stress_job_total(payload),
    )

    def on_phase_start(concurrency: int) -> None:
        reporter.start(
            {"concurrency": concurrency, "phase_seconds": payload.phase_seconds}
        )

    def on_phase(phase: ConcurrencyResult) -> None:
        reporter.done(_phase_row(phase))

    return {"on_phase_start": on_phase_start, "on_phase": on_phase}


def _soak_hooks(payload: SoakRequest, job: Job | None) -> dict[str, Any]:
    if job is None:
        return {}
    reporter = LoadJobReporter(
        job,
        kind="soak",
        meta={
            "provider": payload.provider,
            "model": payload.model,
            "budget_usd": payload.budget_usd,
            "concurrency": payload.concurrency,
            "duration_seconds": payload.duration_seconds,
            "sample_interval": payload.sample_interval,
        },
        rows_key="samples",
        total=soak_job_total(payload),
    )

    def on_sample_start(elapsed: float) -> None:
        reporter.start(
            {"elapsed_seconds": elapsed, "concurrency": payload.concurrency}
        )

    def on_sample(sample: SoakSample) -> None:
        reporter.done(_sample_row(sample))

    return {"on_sample_start": on_sample_start, "on_sample": on_sample}


async def run_stress_from_request(
    payload: StressRequest, job: Job | None = None
) -> dict[str, Any]:
    provider = _metered(
        payload.provider, payload.model, payload.budget_usd, payload.host
    )
    try:
        result = await run_stress_provider(
            provider,
            model=payload.model,
            max_concurrency=payload.max_concurrency,
            phase_seconds=payload.phase_seconds,
            num_predict=MAX_LOAD_PREDICT,
            **_stress_hooks(payload, job),
        )
    except HTTPException:
        raise
    except EvalBudgetExceededError:
        raise
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _stress_payload(payload, result)


async def run_soak_from_request(
    payload: SoakRequest, job: Job | None = None
) -> dict[str, Any]:
    provider = _metered(
        payload.provider, payload.model, payload.budget_usd, payload.host
    )
    try:
        result = await run_soak_provider(
            provider,
            model=payload.model,
            concurrency=payload.concurrency,
            duration_seconds=float(payload.duration_seconds),
            sample_interval=payload.sample_interval,
            num_predict=MAX_LOAD_PREDICT,
            **_soak_hooks(payload, job),
        )
    except HTTPException:
        raise
    except EvalBudgetExceededError:
        raise
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _soak_payload(payload, result)
