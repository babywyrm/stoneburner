"""Bounded stress and soak jobs for the API.

The CLI can ramp to high concurrency or soak for hours. A remote caller gets
one named model, a required budget, and hard caps on concurrency and duration.
No contention mode, no profile YAML, no baselines.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from atomics.api._runners import _provider_for
from atomics.api.models import MAX_LOAD_PREDICT, SoakRequest, StressRequest
from atomics.eval.budget import BudgetMeter, EvalBudget, EvalBudgetExceededError
from atomics.load.soak import SoakResult, run_soak_provider
from atomics.load.stress import StressResult, run_stress_provider
from atomics.providers.base import BaseProvider


def _metered(provider_name: str, model: str, budget_usd: float) -> BaseProvider:
    meter = BudgetMeter(EvalBudget(budget_limit_usd=budget_usd))
    return meter.wrap(_provider_for(provider_name, model))


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
        "phases": [
            {
                "concurrency": phase.concurrency,
                "requests": phase.requests,
                "failed": phase.failed,
                "aggregate_tps": phase.aggregate_tps,
                "avg_latency_ms": phase.avg_latency_ms,
                "p95_latency_ms": phase.p95_latency_ms,
            }
            for phase in result.phases
        ],
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
        "samples": [
            {
                "elapsed_seconds": sample.elapsed_seconds,
                "requests": sample.requests,
                "failed": sample.failed,
                "aggregate_tps": sample.aggregate_tps,
                "avg_latency_ms": sample.avg_latency_ms,
                "p95_latency_ms": sample.p95_latency_ms,
            }
            for sample in result.samples
        ],
    }


async def run_stress_from_request(payload: StressRequest) -> dict[str, Any]:
    provider = _metered(payload.provider, payload.model, payload.budget_usd)
    try:
        result = await run_stress_provider(
            provider,
            model=payload.model,
            max_concurrency=payload.max_concurrency,
            phase_seconds=payload.phase_seconds,
            num_predict=MAX_LOAD_PREDICT,
        )
    except HTTPException:
        raise
    except EvalBudgetExceededError:
        raise
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _stress_payload(payload, result)


async def run_soak_from_request(payload: SoakRequest) -> dict[str, Any]:
    provider = _metered(payload.provider, payload.model, payload.budget_usd)
    try:
        result = await run_soak_provider(
            provider,
            model=payload.model,
            concurrency=payload.concurrency,
            duration_seconds=float(payload.duration_seconds),
            sample_interval=payload.sample_interval,
            num_predict=MAX_LOAD_PREDICT,
        )
    except HTTPException:
        raise
    except EvalBudgetExceededError:
        raise
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _soak_payload(payload, result)
