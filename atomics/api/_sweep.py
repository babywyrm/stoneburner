"""Run a bounded multi-suite sweep for an API request.

The overnight driver lives in `atomics.eval.gauntlet`. This module is only the
trust-model adapter: required budget, named models, no status files, one job
result. Discovering every tag on the host is a CLI flag and stays one.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import HTTPException

from atomics.api._runners import _provider_for
from atomics.api.job_progress import SweepJobReporter, sweep_job_total
from atomics.api.jobs import Job
from atomics.api.models import SweepRequest
from atomics.eval.budget import BudgetMeter, EvalBudget, EvalBudgetExceededError
from atomics.eval.gauntlet import SuiteJobResult, make_suite_runner, run_gauntlet


async def run_sweep_from_request(
    payload: SweepRequest, job: Job | None = None
) -> dict[str, Any]:
    """Run models × suites under one shared dollar ceiling."""
    meter = BudgetMeter(EvalBudget(budget_limit_usd=payload.budget_usd))
    reporter = None
    if job is not None:
        reporter = SweepJobReporter(
            job,
            provider=payload.provider,
            models=list(payload.models),
            suites=list(payload.suites),
            runs=payload.runs,
            budget_usd=payload.budget_usd,
            total=sweep_job_total(payload),
        )

    def provider_factory(model_name: str):
        return meter.wrap(_provider_for(payload.provider, model_name, payload.host))

    judge_host = payload.judge_host
    if judge_host is None and payload.provider == "ollama":
        judge_host = payload.host
    judge_provider = meter.wrap(_provider_for("ollama", payload.judge_model, judge_host))
    inner = make_suite_runner(
        provider_factory=provider_factory,
        judge_provider=judge_provider,
        judge_model=payload.judge_model,
        runs=payload.runs,
        thinking=payload.thinking,
        thinking_budget=None,
        effort=payload.effort,
        reasoning_mode=payload.reasoning_mode,
    )

    async def run_suite(
        *, model: str, suite: str, skip_incapable: bool
    ) -> SuiteJobResult:
        if reporter is not None:
            reporter.start(model, suite)
        result = await inner(model=model, suite=suite, skip_incapable=skip_incapable)
        if reporter is not None:
            reporter.done(result)
        return result

    try:
        results = await run_gauntlet(
            models=payload.models,
            suites=payload.suites,
            run_suite=run_suite,
            skip_incapable=False,
        )
    except HTTPException:
        raise
    except EvalBudgetExceededError:
        raise
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    jobs = [asdict(row) for row in results]
    return {
        "provider": payload.provider,
        "models": list(payload.models),
        "suites": list(payload.suites),
        "runs": payload.runs,
        "budget_usd": payload.budget_usd,
        "ok": sum(1 for row in results if row.ok),
        "fail": sum(1 for row in results if not row.ok),
        "jobs": jobs,
    }
