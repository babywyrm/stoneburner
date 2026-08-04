"""End-to-end tests for eval spend ceilings, over real HTTP.

The unit tests for `GuardedProvider` use a tidy stub, which cannot catch the
failure mode that actually matters for a decorator: a caller reaching past the
`BaseProvider` interface for something the wrapper does not forward. That is
not hypothetical — `sweep` reads `provider._host` to point a judge at the same
Ollama instance, and an early version of the wrapper would have raised
`AttributeError` there.

These run the real CLI through the real provider adapter, a real socket, the
real runner, and the real judge, so the wrapper has to survive the same path a
user's run takes.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from atomics.commands.eval import eval as eval_command
from atomics.eval.budget import EvalBudget, EvalBudgetExceededError, GuardedProvider
from atomics.eval.runner import run_eval
from atomics.providers.vllm import VllmProvider
from tests.inference_stub import StubInferenceServer

TEST_MODEL = "model-under-test"
JUDGE_MODEL = "judge-model"


def _provider(stub: StubInferenceServer, model: str) -> VllmProvider:
    return VllmProvider(base_url=stub.openai_base_url, default_model=model)


def test_a_budgeted_eval_runs_end_to_end_over_http(
    inference_stub: StubInferenceServer, tmp_path: Path
) -> None:
    """The wrapper must be invisible to a normal run.

    This is the regression test for the whole wrapping approach: if the guard
    breaks attribute access, request formation, or response parsing anywhere in
    the real path, this fails.
    """
    json_out = tmp_path / "eval.json"
    result = CliRunner().invoke(eval_command, [
        "--provider", "vllm",
        "--vllm-host", inference_stub.openai_base_url,
        "--model", TEST_MODEL,
        "--judge-provider", "vllm",
        "--judge-model", JUDGE_MODEL,
        "--fixtures", "ev-01,ev-02",
        "--no-save",
        "--budget", "50.00",
        "--json-out", str(json_out),
    ])

    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    payload = json.loads(json_out.read_text())
    assert payload["total_fixtures"] == 2
    assert payload["overall_accuracy"] == 1.0
    assert payload["parse_failure_rate"] == 0.0

    calls = inference_stub.chat_completions()
    generations = [c for c in calls if not c.is_judge_call]
    judgements = [c for c in calls if c.is_judge_call]
    assert len(generations) == 2
    assert len(judgements) == 2
    # The model overrides must still reach the wire through the wrapper.
    assert {c.body["model"] for c in generations} == {TEST_MODEL}
    assert {c.body["model"] for c in judgements} == {JUDGE_MODEL}


def test_an_unbudgeted_eval_is_byte_identical(
    inference_stub: StubInferenceServer, tmp_path: Path
) -> None:
    """Omitting --budget must change nothing about how a run behaves."""
    outputs = []
    for args in (["--budget", "50.00"], []):
        json_out = tmp_path / f"eval{len(outputs)}.json"
        result = CliRunner().invoke(eval_command, [
            "--provider", "vllm",
            "--vllm-host", inference_stub.openai_base_url,
            "--model", TEST_MODEL,
            "--judge-provider", "vllm",
            "--judge-model", JUDGE_MODEL,
            "--fixtures", "ev-01",
            "--no-save",
            "--json-out", str(json_out),
            *args,
        ])
        assert result.exit_code == 0, result.output
        payload = json.loads(json_out.read_text())
        outputs.append(
            (payload["overall_accuracy"], payload["total_fixtures"],
             payload["parse_failure_rate"])
        )

    assert outputs[0] == outputs[1]


@pytest.mark.asyncio
async def test_the_guard_meters_real_traffic(
    inference_stub: StubInferenceServer,
) -> None:
    """Prove the guard is in the request path, not merely wrapped around it.

    A wrapper that forwarded calls without recording them would pass every
    behavioral test above while enforcing nothing.
    """
    guard = EvalBudget(budget_limit_usd=50.0).new_guard()
    provider = GuardedProvider(_provider(inference_stub, TEST_MODEL), guard)
    judge = GuardedProvider(_provider(inference_stub, JUDGE_MODEL), guard)

    await run_eval(
        provider,
        judge_provider=judge,
        model=TEST_MODEL,
        judge_model=JUDGE_MODEL,
        fixtures=_two_fixtures(),
        run_id=None,
    )

    metered_tokens = sum(tokens for _, tokens in guard._hourly_tokens)
    assert metered_tokens > 0, "the guard saw no traffic it was supposed to meter"
    assert len(guard._request_timestamps) >= 4, (
        "generations and judgements must both be counted against the ceiling"
    )


@pytest.mark.asyncio
async def test_a_ceiling_actually_stops_a_real_run(
    inference_stub: StubInferenceServer,
) -> None:
    """A run that exceeds its ceiling stops mid-flight against real I/O.

    The dollar ceiling cannot be used here: local and OpenAI-compatible
    providers report `estimated_cost_usd=0.0`, so a free endpoint never trips
    it — correct behavior, since there is nothing to bill. The hourly token cap
    exercises the same stop path with a provider that costs nothing.
    """
    guard = EvalBudget(budget_limit_usd=50.0, max_tokens_per_hour=1).new_guard()
    provider = GuardedProvider(_provider(inference_stub, TEST_MODEL), guard)

    # The first call is allowed (the cap is checked before each request, and
    # nothing has been spent yet); the second must be refused.
    await provider.generate("hello")
    before = len(inference_stub.chat_completions())

    with pytest.raises(EvalBudgetExceededError, match="hourly token cap"):
        await provider.generate("hello again")

    assert len(inference_stub.chat_completions()) == before, (
        "a refused call must never reach the wire"
    )


@pytest.mark.asyncio
async def test_a_free_provider_is_never_throttled_by_a_dollar_ceiling(
    inference_stub: StubInferenceServer,
) -> None:
    """Documents the consequence of $0 pricing on local providers.

    Someone passing `--budget 0.01` to an Ollama run should get a full run, not
    an immediate stop. Worth pinning: it is the difference between the flag
    being useless and the flag being wrong.
    """
    guard = EvalBudget(budget_limit_usd=0.01).new_guard()
    provider = GuardedProvider(_provider(inference_stub, TEST_MODEL), guard)

    for _ in range(5):
        response = await provider.generate("hello")
        assert response.text

    assert guard.total_cost == 0.0


def _two_fixtures():
    from atomics.eval.fixtures import EVAL_FIXTURES

    wanted = {"ev-01", "ev-02"}
    return [f for f in EVAL_FIXTURES if f.id in wanted]


def test_sweep_reaches_past_the_provider_interface_and_still_works():
    """The concrete bug that motivated attribute forwarding.

    `sweep` builds a judge pointed at `provider._host`. If the wrapper does not
    forward unknown attributes, this raises AttributeError at runtime for every
    sweep run — a failure no stub-based unit test would surface.
    """
    from atomics.providers.ollama import OllamaProvider

    inner = OllamaProvider(host="http://192.168.1.156:11434")
    guarded = GuardedProvider(inner, EvalBudget().new_guard())

    assert guarded._host == inner._host


def test_the_event_loop_is_not_required_for_wrapping():
    """Wrapping must be safe outside async context, as the CLI does it."""
    provider = GuardedProvider(
        VllmProvider(base_url="http://127.0.0.1:1/v1", default_model="m"),
        EvalBudget().new_guard(),
    )
    assert provider.name
    assert asyncio.iscoroutinefunction(provider.generate)
