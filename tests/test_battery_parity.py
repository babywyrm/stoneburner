"""CLI `battery run` and `POST /batteries` must hand the suite runners the same work.

Both sides build their own request per step, and that is where they drifted:
fixture scope, runs, thinking, the save switch, and temperature each differed
once. This runs the real CLI command and the real API path with each runner
replaced by a recorder, and compares what the runner would have received.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any
from unittest.mock import patch

import pytest

from atomics.api._battery import _step_to_eval
from atomics.api._runners import run_eval_suite
from atomics.api.models import BatteryRequest
from atomics.commands.battery import _with_runs, invoke_atomics
from atomics.eval.batteries import BATTERIES, step_args, visible_steps

# (CLI import site, API import site). The CLI imports lazily from the package.
RUNNERS = {
    "adversarial": (
        "atomics.eval.adversarial.runner.run_adversarial",
        "atomics.api._runners.run_adversarial",
    ),
    "redblue": ("atomics.eval.redblue.runner.run_redblue", "atomics.api._runners.run_redblue"),
    "refusal": ("atomics.eval.refusal.run_refusal", "atomics.api._runners.run_refusal"),
    "codereview": (
        "atomics.eval.codereview.run_codereview",
        "atomics.api._runners.run_codereview",
    ),
    "toolcall": (
        "atomics.commands.toolcall.run_toolcall_suite",
        "atomics.api._runners.run_toolcall_suite",
    ),
}
COMPARED = (
    "fixtures",
    "runs",
    "temperature",
    "thinking",
    "effort",
    "reasoning_mode",
    "channel",
    "model",
    "judge_model",
)
OPTIONS = [
    {},
    {"runs": 2, "temperature": 0.0, "thinking": True, "effort": "low"},
]
STEPS = [
    (battery, step)
    for battery in BATTERIES
    for step in visible_steps(battery, provider="ollama")
    if step.suite in RUNNERS
]


class _CapturedError(Exception):
    """Stops the run at the runner boundary; neither side catches it as a suite error."""


def _resolve(target: str) -> Any:
    module, name = target.rsplit(".", 1)
    return getattr(__import__(module, fromlist=[name]), name)


def _recorder(target: str, sink: dict[str, Any]) -> Any:
    signature = inspect.signature(_resolve(target))

    async def record(*args: Any, **kwargs: Any) -> None:
        bound = signature.bind_partial(*args, **kwargs).arguments
        defaults = {
            name: param.default
            for name, param in signature.parameters.items()
            if param.default is not inspect.Parameter.empty
        }
        for key in COMPARED:
            if key in signature.parameters:
                value = bound.get(key, defaults.get(key))
                if key == "fixtures" and value is not None:
                    value = sorted(f.id for f in value)
                sink[key] = value
        raise _CapturedError

    return record


def _cli_call(battery_id: str, step: Any, options: dict[str, Any]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    argv = step_args(
        step,
        model="m",
        provider="ollama",
        thinking_flag="--thinking" if options.get("thinking") else "--no-thinking",
        judge_model="judge",
        effort=options.get("effort"),
        temperature=options.get("temperature"),
    )
    _with_runs(argv, step, options.get("runs"))
    target = RUNNERS[step.suite][0]
    with patch(target, _recorder(target, seen)):
        try:
            invoke_atomics(argv)
        except _CapturedError:
            pass
    return seen


def _api_call(battery_id: str, step: Any, options: dict[str, Any]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    payload = BatteryRequest(
        name=battery_id,
        provider="ollama",
        model="m",
        judge_model="judge",
        budget_usd=1.0,
        **options,
    )
    target = RUNNERS[step.suite][1]
    with patch(target, _recorder(target, seen)):
        try:
            asyncio.run(run_eval_suite(_step_to_eval(payload, step)))
        except _CapturedError:
            pass
    return seen


@pytest.mark.parametrize("options", OPTIONS, ids=["defaults", "overrides"])
@pytest.mark.parametrize(
    ("battery", "step"),
    STEPS,
    ids=[f"{battery.id}-{step.suite}" for battery, step in STEPS],
)
def test_cli_and_api_hand_the_runner_the_same_work(battery, step, options):
    cli = _cli_call(battery.id, step, options)
    api = _api_call(battery.id, step, options)
    assert cli, "the CLI never reached the runner"
    assert api == cli


def test_every_runner_backed_suite_is_covered():
    suites = {step.suite for battery in BATTERIES for step in battery.steps}
    assert suites - {"provider-test", "qa"} <= set(RUNNERS)
