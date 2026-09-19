"""Run a named battery as one metered API job.

The CLI `atomics battery run` composes suites in-process. This module is the
API trust-model adapter: required budget, one job, ordered steps. provider-test
runs through the discovery probe (fixed 2+2, no spend amp); qa runs through
`run_qa_suite` on the battery's fixture file or profile. archreview needs a
repo pack and stays on the CLI.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from atomics.api._discovery import run_provider_test
from atomics.api._runners import run_eval_suite
from atomics.api.jobs import Job
from atomics.api.models import BatteryRequest, EvalRequest, ProviderTestRequest
from atomics.benchmark.qa_runner import load_qa_suite, run_qa_suite
from atomics.eval.batteries import get_battery, visible_steps
from atomics.eval.budget import EvalBudgetExceededError


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


async def _run_provider_test_step(payload: BatteryRequest) -> dict[str, Any]:
    result = await run_provider_test(
        ProviderTestRequest(
            provider=payload.provider,
            model=payload.model,
            host=payload.host,
            thinking=payload.thinking,
            effort=payload.effort,
            reasoning_mode=payload.reasoning_mode,
        )
    )
    return {
        "suite": "provider-test",
        "ok": bool(result.get("ok")),
        "health": result.get("health"),
        "error": result.get("error"),
        "cost_usd": result.get("cost_usd", 0.0),
    }


async def _run_qa_step(payload: BatteryRequest, step: Any) -> dict[str, Any]:
    if not step.qa_file and not payload.profile:
        return {"suite": "qa", "skipped": True, "reason": "no qa_file or profile"}
    if payload.profile:
        from atomics.load.profiles import load_profile

        profile = load_profile(payload.profile)
        suite = await run_qa_suite(
            model="",
            host="",
            fixtures=load_qa_suite(step.qa_file)[2] if step.qa_file else [],
            profile=profile,
            thinking=payload.thinking,
            effort=payload.effort,
        )
    else:
        file_model, file_host, fixtures = load_qa_suite(step.qa_file)
        suite = await run_qa_suite(
            model=payload.model or file_model,
            host=payload.host or file_host,
            fixtures=fixtures,
            thinking=payload.thinking,
            effort=payload.effort,
        )
    return {
        "suite": "qa",
        "ok": suite.failed == 0 and suite.errors == 0,
        "passed": suite.passed,
        "failed": suite.failed,
        "errors": suite.errors,
        "total": suite.total,
    }


async def run_battery_from_request(
    payload: BatteryRequest, job: Job | None = None
) -> dict[str, Any]:
    """Run the battery's steps in order under one budget."""
    battery = get_battery(payload.name)
    steps_out: list[dict[str, Any]] = []
    visible = visible_steps(battery, provider=payload.provider, profile=payload.profile)
    all_steps = (*visible, *battery.optional_steps)
    for step in all_steps:
        try:
            if step.suite == "provider-test":
                steps_out.append(await _run_provider_test_step(payload))
                continue
            if step.suite == "qa":
                steps_out.append(await _run_qa_step(payload, step))
                continue
            if step.suite == "archreview":
                steps_out.append({"suite": step.suite, "skipped": True, "reason": "repo pack"})
                continue
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
