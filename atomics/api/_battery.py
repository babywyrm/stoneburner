"""Run a named battery as one metered API job.

The CLI `atomics battery run` composes suites in-process. This module is the
API trust-model adapter: required budget, one job, ordered steps. provider-test
and qa are not eval suites and are recorded as skipped in v1; archreview needs
a repo pack and stays on the CLI.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from atomics.api._runners import run_eval_suite
from atomics.api.jobs import Job
from atomics.api.models import BatteryRequest, EvalRequest
from atomics.eval.batteries import get_battery, visible_steps
from atomics.eval.budget import EvalBudgetExceededError

# Not eval suites: provider-test and qa have their own paths; archreview needs
# a repo pack. v1 records these as skipped rather than running them.
_SKIP = frozenset({"provider-test", "qa", "archreview"})


def _step_channel(step: Any) -> str | None:
    extra = list(step.extra)
    if "--channel" in extra:
        return extra[extra.index("--channel") + 1]
    return None


def _step_to_eval(payload: BatteryRequest, step: Any) -> EvalRequest:
    return EvalRequest(
        suite=step.suite,
        provider=payload.provider,
        model=payload.model,
        judge_model=payload.judge_model,
        host=payload.host,
        judge_host=payload.judge_host,
        budget_usd=payload.budget_usd,
        thinking=payload.thinking,
        effort=payload.effort,
        reasoning_mode=payload.reasoning_mode,
        runs=payload.runs,
        fixtures=list(step.fixtures) or None,
        channel=_step_channel(step),
    )


async def run_battery_from_request(
    payload: BatteryRequest, job: Job | None = None
) -> dict[str, Any]:
    """Run the battery's eval-suite steps in order under one budget."""
    battery = get_battery(payload.name)
    steps_out: list[dict[str, Any]] = []
    for step in visible_steps(battery, provider=payload.provider, profile=payload.profile):
        if step.suite in _SKIP:
            steps_out.append({"suite": step.suite, "skipped": True})
            continue
        try:
            result = await run_eval_suite(_step_to_eval(payload, step), job=job)
        except EvalBudgetExceededError:
            raise
        except HTTPException:
            raise
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(
                status_code=400, detail=f"{payload.name}:{step.suite}: {exc}"
            ) from exc
        steps_out.append({"suite": step.suite, "ok": True, "result": result})
    return {
        "battery": payload.name,
        "provider": payload.provider,
        "model": payload.model,
        "budget_usd": payload.budget_usd,
        "steps": steps_out,
    }
