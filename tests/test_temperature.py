"""Temperature plumbing across providers + deterministic judge scoring.

Verifies that an explicit temperature is forwarded to each backend where it is
valid, and is correctly withheld for the two cases that reject/forbid it:
OpenAI reasoning models and Claude with extended thinking enabled.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from atomics.providers.base import BaseProvider, ProviderResponse


# ── Ollama ──────────────────────────────────────────────────────────────────
def _ollama_response():
    r = MagicMock()
    r.status_code = 200
    r.raise_for_status = MagicMock()
    r.json.return_value = {
        "response": "ok",
        "eval_count": 5,
        "prompt_eval_count": 3,
        "eval_duration": 100_000_000,
    }
    return r


@pytest.mark.asyncio
async def test_ollama_forwards_temperature():
    from atomics.providers.ollama import OllamaProvider

    client = AsyncMock()
    client.post = AsyncMock(return_value=_ollama_response())
    provider = OllamaProvider(host="http://fake:11434", client=client)
    await provider.generate("hi", temperature=0.0)
    body = client.post.call_args.kwargs["json"]
    assert body["options"]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_ollama_omits_temperature_when_none():
    from atomics.providers.ollama import OllamaProvider

    client = AsyncMock()
    client.post = AsyncMock(return_value=_ollama_response())
    provider = OllamaProvider(host="http://fake:11434", client=client)
    await provider.generate("hi")
    body = client.post.call_args.kwargs["json"]
    assert "temperature" not in body.get("options", {})


# ── vLLM ────────────────────────────────────────────────────────────────────
def _vllm_response():
    r = MagicMock()
    r.status_code = 200
    r.raise_for_status = MagicMock()
    r.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    }
    return r


@pytest.mark.asyncio
async def test_vllm_forwards_temperature():
    from atomics.providers.vllm import VllmProvider

    client = AsyncMock()
    client.post = AsyncMock(return_value=_vllm_response())
    provider = VllmProvider(base_url="http://fake:8000/v1", client=client)
    await provider.generate("hi", temperature=0.2)
    body = client.post.call_args.kwargs["json"]
    assert body["temperature"] == 0.2


@pytest.mark.asyncio
async def test_vllm_omits_temperature_when_none():
    from atomics.providers.vllm import VllmProvider

    client = AsyncMock()
    client.post = AsyncMock(return_value=_vllm_response())
    provider = VllmProvider(base_url="http://fake:8000/v1", client=client)
    await provider.generate("hi")
    body = client.post.call_args.kwargs["json"]
    assert "temperature" not in body


# ── OpenAI (chat completions) ───────────────────────────────────────────────
class _OAIUsage:
    prompt_tokens = 10
    completion_tokens = 5
    completion_tokens_details = None


class _OAIChoice:
    class _M:
        content = "ok"

    message = _M()


class _OAIResp:
    choices = [_OAIChoice()]
    usage = _OAIUsage()

    def model_dump(self):
        return {}


class _CapturingCompletions:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return _OAIResp()


def _openai_provider():
    from atomics.providers.openai import OpenAIProvider

    comps = _CapturingCompletions()
    client = type("C", (), {"chat": type("Chat", (), {"completions": comps})()})()
    return OpenAIProvider(api_key="fake", client=client), comps


@pytest.mark.asyncio
async def test_openai_forwards_temperature_for_chat_model():
    provider, comps = _openai_provider()
    await provider.generate("hi", model="gpt-4o", temperature=0.0)
    assert comps.kwargs["temperature"] == 0.0


@pytest.mark.asyncio
async def test_openai_omits_temperature_for_reasoning_model():
    provider, comps = _openai_provider()
    await provider.generate("hi", model="o4-mini", temperature=0.0)
    # o-series rejects an explicit temperature — must not be sent.
    assert "temperature" not in comps.kwargs


@pytest.mark.asyncio
async def test_openai_gpt_5_6_uses_reasoning_model_parameters():
    provider, comps = _openai_provider()
    await provider.generate("hi", model="gpt-5.6", max_tokens=32, temperature=0.0)
    assert comps.kwargs["max_completion_tokens"] == 256
    assert "max_tokens" not in comps.kwargs
    assert "temperature" not in comps.kwargs


@pytest.mark.asyncio
async def test_openai_responses_gpt_5_6_dated_alias_uses_reasoning_parameters():
    from atomics.providers.openai import OpenAIProvider

    class Auth:
        async def get_headers(self):
            return {"Authorization": "Bearer test"}

    class Responses:
        def __init__(self):
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return type(
                "Response",
                (),
                {
                    "status": "completed",
                    "output_text": "ok",
                    "output": [],
                    "usage": type(
                        "Usage",
                        (),
                        {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "output_tokens_details": None,
                        },
                    )(),
                },
            )()

    responses = Responses()
    client = type("C", (), {"responses": responses, "api_key": "oauth-managed"})()
    provider = OpenAIProvider(client=client, auth=Auth())

    await provider.generate(
        "hi",
        model="gpt-5.6-2026-06-01",
        max_tokens=32,
        temperature=0.0,
    )

    assert responses.kwargs["max_output_tokens"] == 256
    assert "temperature" not in responses.kwargs


@pytest.mark.asyncio
async def test_openai_omits_temperature_when_none():
    provider, comps = _openai_provider()
    await provider.generate("hi", model="gpt-4o")
    assert "temperature" not in comps.kwargs


# ── Claude ──────────────────────────────────────────────────────────────────
class _ClaudeResp:
    class _Text:
        def __init__(self, t):
            self.text = t
            self.type = "text"

    class _Usage:
        input_tokens = 10
        output_tokens = 20

    def __init__(self):
        self.content = [self._Text("ok")]
        self.usage = self._Usage()

    def model_dump(self):
        return {}


class _CapturingMessages:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return _ClaudeResp()


def _claude_provider():
    from atomics.providers.claude import ClaudeProvider

    msgs = _CapturingMessages()
    client = type("C", (), {"messages": msgs})()
    return ClaudeProvider(api_key="fake", client=client), msgs


@pytest.mark.asyncio
async def test_claude_forwards_temperature_without_thinking():
    provider, msgs = _claude_provider()
    await provider.generate("hi", model="claude-sonnet-4-6", temperature=0.0)
    assert msgs.kwargs["extra_body"] == {"temperature": 0.0}


@pytest.mark.asyncio
async def test_claude_omits_temperature_with_thinking():
    provider, msgs = _claude_provider()
    await provider.generate(
        "hi",
        model="claude-sonnet-4-6",
        thinking=True,
        temperature=0.0,
    )
    # Anthropic requires temperature=1 with extended thinking, so we must not
    # send our explicit value.
    assert "extra_body" not in msgs.kwargs
    assert "thinking" in msgs.kwargs


# ── Bedrock ─────────────────────────────────────────────────────────────────
class _CapturingBedrock:
    def __init__(self):
        self.kwargs = None

    def converse(self, **kwargs):
        self.kwargs = kwargs
        return {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }


@pytest.mark.asyncio
async def test_bedrock_forwards_temperature():
    from atomics.providers.bedrock import BedrockProvider

    client = _CapturingBedrock()
    provider = BedrockProvider(client=client)
    await provider.generate("hi", temperature=0.0)
    assert client.kwargs["inferenceConfig"]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_bedrock_omits_temperature_when_none():
    from atomics.providers.bedrock import BedrockProvider

    client = _CapturingBedrock()
    provider = BedrockProvider(client=client)
    await provider.generate("hi")
    assert "temperature" not in client.kwargs["inferenceConfig"]


# ── Tool-calling path ───────────────────────────────────────────────────────
_TOOLS = [{"name": "read_file", "description": "d", "parameters": {"type": "object"}}]


@pytest.mark.asyncio
@pytest.mark.parametrize("temperature", [0.0, None])
async def test_ollama_tools_forward_temperature(temperature):
    from atomics.providers.ollama import OllamaProvider

    reply = MagicMock()
    reply.raise_for_status = MagicMock()
    reply.json.return_value = {"message": {"content": "ok"}, "eval_count": 1}
    client = AsyncMock()
    client.post = AsyncMock(return_value=reply)
    provider = OllamaProvider(host="http://fake:11434", client=client)
    await provider.generate_with_tools("hi", tools=_TOOLS, temperature=temperature)
    assert client.post.call_args.kwargs["json"]["options"].get("temperature") == temperature


@pytest.mark.asyncio
@pytest.mark.parametrize("temperature", [0.0, None])
async def test_openai_compatible_tools_forward_temperature(temperature):
    from atomics.providers.vllm import VllmProvider

    client = AsyncMock()
    client.post = AsyncMock(return_value=_vllm_response())
    provider = VllmProvider(base_url="http://fake:8000/v1", client=client)
    await provider.generate_with_tools("hi", tools=_TOOLS, temperature=temperature)
    assert client.post.call_args.kwargs["json"].get("temperature") == temperature


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "sent"), [("gpt-4o", 0.0), ("o4-mini", None)])
async def test_openai_tools_forward_temperature_except_reasoning(model, sent):
    provider, comps = _openai_provider()
    await provider.generate_with_tools("hi", tools=_TOOLS, model=model, temperature=0.0)
    assert comps.kwargs.get("temperature") == sent


@pytest.mark.asyncio
async def test_claude_tools_forward_temperature():
    provider, msgs = _claude_provider()
    await provider.generate_with_tools(
        "hi", tools=_TOOLS, model="claude-sonnet-4-6", temperature=0.0
    )
    assert msgs.kwargs["extra_body"] == {"temperature": 0.0}


@pytest.mark.asyncio
@pytest.mark.parametrize("tools", [False, True])
async def test_claude_request_through_the_real_sdk(tools):
    """The fake client takes any kwarg; the real SDK raises TypeError on one it dropped."""
    anthropic = pytest.importorskip("anthropic")
    import httpx2

    from atomics.providers.claude import ClaudeProvider

    sent: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = anthropic.AsyncAnthropic(
        api_key="fake",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    provider = ClaudeProvider(api_key="fake", client=client)
    if tools:
        await provider.generate_with_tools(
            "hi", tools=_TOOLS, model="claude-sonnet-4-6", temperature=0.0, effort="low"
        )
    else:
        await provider.generate(
            "hi", model="claude-sonnet-4-6", temperature=0.0, effort="low", thinking=False
        )
    await provider.aclose()
    assert sent[0]["temperature"] == 0.0
    assert sent[0]["output_config"] == {"effort": "low"}


class _ToolRecorder(BaseProvider):
    supports_tools = True

    def __init__(self) -> None:
        self.kwargs: dict = {}

    @property
    def name(self) -> str:
        return "rec"

    async def generate(self, prompt: str, **kwargs: object) -> ProviderResponse:  # type: ignore[override]
        raise AssertionError("tools path expected")

    async def generate_with_tools(self, prompt: str, **kwargs: object) -> ProviderResponse:  # type: ignore[override]
        self.kwargs = kwargs
        return ProviderResponse(
            text="",
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            model="m",
            latency_ms=1.0,
            estimated_cost_usd=0.0,
        )

    async def health_check(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_budget_and_trace_wrappers_pass_tool_temperature_through() -> None:
    from rich.console import Console

    from atomics.eval.budget import EvalBudget, GuardedProvider
    from atomics.providers import trace

    rec = _ToolRecorder()
    guarded = GuardedProvider(rec, EvalBudget(budget_limit_usd=1.0).new_guard())
    trace.enable(Console(record=True, width=120))
    try:
        await trace.traced(guarded).generate_with_tools("hi", tools=_TOOLS, temperature=0.0)
    finally:
        trace.disable()
    assert rec.kwargs["temperature"] == 0.0


# ── Judge passes temperature=0.0 ────────────────────────────────────────────
class _CapturingJudgeProvider:
    """Records the temperature the judge requests; returns a parseable rubric."""

    name = "fake-judge"

    def __init__(self):
        self.temperature = "unset"

    async def generate(
        self,
        prompt,
        *,
        system="",
        model=None,
        max_tokens=1024,
        thinking=None,
        thinking_budget=None,
        temperature=None,
        **_kwargs,
    ):
        self.temperature = temperature
        return ProviderResponse(
            text="ACCURACY: 4\nCOMPLETENESS: 3\nFORMAT: 3\nRATIONALE: solid.",
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            model="fake-judge",
            latency_ms=1.0,
            estimated_cost_usd=0.0,
        )

    async def health_check(self):
        return True


@pytest.mark.asyncio
async def test_quality_judge_requests_greedy_decoding():
    from atomics.eval.judge import score_response

    judge = _CapturingJudgeProvider()
    result = await score_response("q", "a", judge_provider=judge)
    assert judge.temperature == 0.0
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_resistance_judge_requests_greedy_decoding():
    from atomics.eval.adversarial.scorer import score_resistance

    judge = _CapturingJudgeProvider()
    judge_text = "RESISTANCE: 10\nLABEL: resisted\nRATIONALE: refused."

    async def _gen(
        prompt,
        *,
        system="",
        model=None,
        max_tokens=1024,
        thinking=None,
        thinking_budget=None,
        temperature=None,
        **_kwargs,
    ):
        judge.temperature = temperature
        return ProviderResponse(
            text=judge_text,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            model="fake-judge",
            latency_ms=1.0,
            estimated_cost_usd=0.0,
        )

    judge.generate = _gen
    result = await score_resistance(
        "p",
        "r",
        attack_goal="g",
        resistance_criteria=["c"],
        judge_provider=judge,
    )
    assert judge.temperature == 0.0
    assert result.label == "resisted"


# ── Suites: an opt-in --temperature on qa, toolcall, and battery ────────────
class _ToolcallRecorder:
    supports_tools = True
    name = "rec"

    def __init__(self) -> None:
        self.temperatures: list[object] = []

    @staticmethod
    def _resp(calls=()):
        return ProviderResponse(
            text="no",
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            model="m",
            latency_ms=1.0,
            estimated_cost_usd=0.0,
            tool_calls=calls,
        )

    async def generate(self, prompt, **kwargs):
        self.temperatures.append(kwargs.get("temperature"))
        return self._resp()

    async def generate_with_tools(self, prompt, *, tools, **kwargs):
        from atomics.eval.toolcall.catalog import PROBE_PROMPT
        from atomics.providers.toolcalls import ToolCall

        self.temperatures.append(kwargs.get("temperature"))
        if prompt == PROBE_PROMPT:
            return self._resp((ToolCall(name="get_weather", arguments={"city": "Paris"}),))
        return self._resp()


@pytest.mark.asyncio
async def test_toolcall_suite_pins_probe_tools_and_prose_to_one_temperature():
    from atomics.eval.toolcall.fixtures import select_fixtures
    from atomics.eval.toolcall.runner import run_toolcall_suite

    provider = _ToolcallRecorder()
    summary = await run_toolcall_suite(
        provider=provider,  # type: ignore[arg-type]
        model="m",
        judge_provider=None,
        fixtures=select_fixtures(ids=["tc-01"]),
        temperature=0.0,
    )
    assert provider.temperatures == [0.0, 0.0, 0.0]
    assert summary.to_dict()["temperature"] == 0.0


def test_toolcall_cli_forwards_temperature(monkeypatch):
    from types import SimpleNamespace

    from click.testing import CliRunner

    from atomics.cli import cli
    from atomics.eval.toolcall.runner import ToolCallSummary

    seen: dict = {}

    async def fake_run(**kwargs):
        seen.update(kwargs)
        return ToolCallSummary(
            run_id="r",
            provider="ollama",
            model="x",
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:00:01+00:00",
            tool_capable=False,
        )

    monkeypatch.setattr(
        "atomics.commands.toolcall._make_provider",
        lambda *_a, **_k: SimpleNamespace(name="ollama", default_model="x"),
    )
    monkeypatch.setattr("atomics.commands.toolcall.run_toolcall_suite", fake_run)
    result = CliRunner().invoke(
        cli, ["toolcall", "-p", "ollama", "-m", "x", "--no-save", "--temperature", "0"]
    )
    assert result.exit_code == 0, result.output
    assert seen["temperature"] == 0.0


@pytest.mark.parametrize(
    "command",
    [
        "toolcall",
        "qa",
        "battery",
        "eval",
        "labcompare",
        "adversarial",
        "redblue",
        "refusal",
        "codereview",
    ],
)
def test_temperature_outside_zero_to_two_is_rejected(command):
    from click.testing import CliRunner

    from atomics.cli import cli

    argv = {
        "toolcall": ["toolcall", "-p", "ollama", "-m", "x", "--no-save"],
        "qa": ["qa", "--file", "qa/examples/ctf-solvability.yaml", "-m", "x"],
        "battery": ["battery", "run", "desk-pass", "-m", "x"],
        "labcompare": ["labcompare", "--host", "a=http://localhost:1", "--models", "x"],
    }.get(command, [command, "-m", "x", "--no-save"])
    result = CliRunner().invoke(cli, [*argv, "--temperature", "2.5"])
    assert result.exit_code == 2
    assert "not in the range" in result.output


@pytest.mark.asyncio
@pytest.mark.parametrize("temperature", [0.0, None])
async def test_raw_qa_sends_temperature_only_when_set(temperature):
    from unittest.mock import patch

    from atomics.benchmark.qa_runner import QAFixture, run_qa_suite

    reply = MagicMock()
    reply.raise_for_status = MagicMock()
    reply.json.return_value = {"response": "ok"}
    post = AsyncMock(return_value=reply)
    with patch("httpx.AsyncClient.post", post):
        await run_qa_suite(
            "m", "http://h", [QAFixture(id="t", prompt="q")], temperature=temperature
        )
    assert post.call_args.kwargs["json"]["options"].get("temperature") == temperature


def test_qa_rejects_temperature_with_a_profile():
    from click.testing import CliRunner

    from atomics.cli import cli

    result = CliRunner().invoke(
        cli,
        [
            "qa",
            "--file",
            "qa/examples/app-gate-guardrails.yaml",
            "--profile",
            "profiles/examples/ollama-with-system-prompt.yaml",
            "--temperature",
            "0",
        ],
    )
    assert result.exit_code == 2
    assert "profile" in result.output


@pytest.mark.parametrize(
    "suite", ["qa", "toolcall", "adversarial", "redblue", "refusal", "codereview"]
)
def test_battery_pins_every_suite_that_samples_the_model(suite):
    from atomics.eval.batteries import BatteryStep, step_args

    step = BatteryStep(suite=suite, purpose="p", qa_file="qa/x.yaml")
    assert step_args(step, model="m", temperature=0.0)[-2:] == ["--temperature", "0.0"]


def test_battery_leaves_temperature_off_where_it_does_not_apply():
    from atomics.eval.batteries import BatteryStep, step_args

    def args(suite: str, **kwargs) -> list[str]:
        step = BatteryStep(suite=suite, purpose="p", qa_file="qa/x.yaml", repo="r")
        return step_args(step, model="m", temperature=0.0, **kwargs)

    assert "--temperature" not in args("provider-test")
    assert "--temperature" not in args("archreview")
    assert "--temperature" not in args("qa", profile="profiles/local/gate.yaml")
    assert "--temperature" not in step_args(BatteryStep(suite="toolcall", purpose="p"), model="m")


@pytest.mark.parametrize("battery", ["desk-pass", "blue-capability", "threat-model"])
def test_battery_run_forwards_temperature(monkeypatch, battery):
    from click.testing import CliRunner

    from atomics.cli import cli

    seen: list[list[str]] = []
    monkeypatch.setattr(
        "atomics.commands.battery.invoke_atomics", lambda args: seen.append(args) or 0
    )
    result = CliRunner().invoke(
        cli,
        ["battery", "run", battery, "-m", "x", "--judge-model", "j", "--temperature", "0"],
    )
    assert result.exit_code == 0, result.output
    sampling = {"qa", "toolcall", "adversarial", "redblue", "refusal", "codereview"}
    assert any(args[0] in sampling for args in seen)
    for args in seen:
        assert ("--temperature" in args) == (args[0] in sampling), args


class _TemperatureRecorder(BaseProvider):
    def __init__(self, text: str = "I can't help with that.") -> None:
        self.text = text
        self.temperatures: list[object] = []

    @property
    def name(self) -> str:
        return "rec"

    async def generate(self, prompt: str, **kwargs: object) -> ProviderResponse:  # type: ignore[override]
        self.temperatures.append(kwargs.get("temperature"))
        return ProviderResponse(
            text=self.text,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            model="m",
            latency_ms=1.0,
            estimated_cost_usd=0.0,
        )

    async def health_check(self) -> bool:
        return True


def _judged_suite(suite: str):
    if suite == "adversarial":
        from atomics.eval.adversarial import ALL_FIXTURES
        from atomics.eval.adversarial.runner import run_adversarial

        return run_adversarial, ALL_FIXTURES[:1]
    if suite == "redblue":
        from atomics.eval.redblue.fixtures import ALL_FIXTURES
        from atomics.eval.redblue.runner import run_redblue

        return run_redblue, ALL_FIXTURES[:1]
    if suite == "refusal":
        from atomics.eval.refusal.fixtures import REFUSAL_FIXTURES
        from atomics.eval.refusal.runner import run_refusal

        return run_refusal, REFUSAL_FIXTURES[:1]
    from atomics.eval.codereview.fixtures import SECURE_CODE_FIXTURES
    from atomics.eval.codereview.runner import run_codereview

    return run_codereview, SECURE_CODE_FIXTURES[:1]


@pytest.mark.asyncio
@pytest.mark.parametrize("suite", ["adversarial", "redblue", "refusal", "codereview"])
async def test_judged_suites_pin_the_model_and_keep_the_judge_at_zero(suite):
    run, fixtures = _judged_suite(suite)
    target = _TemperatureRecorder()
    judge = _TemperatureRecorder(text="{}")
    summary = await run(
        target, judge_provider=judge, model="m", fixtures=list(fixtures), temperature=0.7
    )
    assert target.temperatures and set(target.temperatures) == {0.7}
    assert set(judge.temperatures) <= {0.0}
    assert summary.to_dict()["temperature"] == 0.7


_RUNNER_PATHS = {
    "eval": "atomics.eval.runner.run_eval",
    "adversarial": "atomics.eval.adversarial.runner.run_adversarial",
    "redblue": "atomics.eval.redblue.runner.run_redblue",
    "refusal": "atomics.eval.refusal.run_refusal",
    "codereview": "atomics.eval.codereview.run_codereview",
}
_COMMAND_MODULES = {"eval": "atomics.commands.eval"}


@pytest.mark.parametrize("suite", sorted(_RUNNER_PATHS))
def test_judged_commands_forward_temperature(monkeypatch, suite):
    from click.testing import CliRunner

    from atomics.cli import cli

    seen: dict = {}

    async def fake_run(*_args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(
        f"{_COMMAND_MODULES.get(suite, f'atomics.commands.security.cmd_{suite}')}._make_provider",
        lambda *_a, **_k: _TemperatureRecorder(),
    )
    monkeypatch.setattr(_RUNNER_PATHS[suite], fake_run)
    CliRunner().invoke(
        cli, [suite, "-m", "x", "--judge-model", "j", "--no-save", "--temperature", "0.5"]
    )
    assert seen.get("temperature") == 0.5


# ── sweep: the eval suite, the gauntlet, and the command ────────────────────
@pytest.mark.asyncio
async def test_eval_suite_pins_the_model_and_keeps_the_judge_at_zero():
    from atomics.eval.fixtures import EVAL_FIXTURES
    from atomics.eval.runner import run_eval

    target = _TemperatureRecorder()
    judge = _TemperatureRecorder(text="{}")
    summary = await run_eval(
        target, judge_provider=judge, model="m", fixtures=EVAL_FIXTURES[:1], temperature=0.7
    )
    assert target.temperatures and set(target.temperatures) == {0.7}
    assert set(judge.temperatures) <= {0.0}
    assert summary.to_dict()["temperature"] == 0.7


_SWEEP_RUNNERS = {
    "eval": "atomics.eval.runner.run_eval",
    "redblue": "atomics.eval.redblue.runner.run_redblue",
    "refusal": "atomics.eval.refusal.run_refusal",
    "toolcall": "atomics.eval.toolcall.runner.run_toolcall_suite",
    "codereview": "atomics.eval.codereview.run_codereview",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("suite", sorted(_SWEEP_RUNNERS))
async def test_gauntlet_forwards_temperature_to_every_suite(monkeypatch, suite):
    from atomics.eval.gauntlet import make_suite_runner

    seen: dict = {}

    async def fake_run(*_args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(_SWEEP_RUNNERS[suite], fake_run)
    run_suite = make_suite_runner(
        provider_factory=lambda _model: _TemperatureRecorder(),
        judge_provider=_TemperatureRecorder(),
        judge_model=None,
        runs=1,
        thinking=None,
        thinking_budget=None,
        temperature=0.3,
    )
    result = await run_suite(model="m", suite=suite, skip_incapable=False)
    assert not result.ok
    assert seen.get("temperature") == 0.3


@pytest.mark.asyncio
async def test_model_sweep_forwards_temperature(monkeypatch):
    from atomics.benchmark.sweep import run_model_sweep

    seen: dict = {}

    async def fake_run_eval(*_args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after capture")

    monkeypatch.setattr("atomics.benchmark.sweep.run_eval", fake_run_eval)
    await run_model_sweep(
        provider_factory=lambda _model: _TemperatureRecorder(),
        judge_provider=_TemperatureRecorder(),
        models=["m"],
        temperature=0.3,
    )
    assert seen.get("temperature") == 0.3


@pytest.mark.parametrize(
    ("suites", "target"),
    [
        ("eval", "atomics.benchmark.sweep.run_model_sweep"),
        ("redblue,toolcall", "atomics.eval.gauntlet.make_suite_runner"),
    ],
)
def test_sweep_cli_forwards_temperature(monkeypatch, suites, target):
    from click.testing import CliRunner

    from atomics.cli import cli

    seen: dict = {}

    def capture(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(
        "atomics.commands.sweep._make_provider", lambda *_a, **_k: _TemperatureRecorder()
    )
    monkeypatch.setattr(target, capture)
    CliRunner().invoke(
        cli, ["sweep", "--models", "a,b", "--suites", suites, "--temperature", "0.3"]
    )
    assert seen.get("temperature") == 0.3


@pytest.mark.parametrize(
    ("suite", "runner"),
    [
        ("eval", "atomics.eval.runner.run_eval"),
        ("redblue", "atomics.eval.redblue.runner.run_redblue"),
    ],
)
def test_labcompare_pins_the_quality_suite(monkeypatch, suite, runner):
    import asyncio

    from click.testing import CliRunner

    from atomics.cli import cli

    seen: dict = {}

    async def fake_run(*_args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after capture")

    def fake_sync(**kwargs):
        asyncio.run(kwargs["quality_fn"](_TemperatureRecorder(), "http://judge", "j", "m"))

    monkeypatch.setattr(
        "atomics.commands.labcompare._make_provider", lambda *_a, **_k: _TemperatureRecorder()
    )
    monkeypatch.setattr("atomics.commands.labcompare._run_labcompare_sync", fake_sync)
    monkeypatch.setattr(runner, fake_run)
    CliRunner().invoke(
        cli,
        [
            "labcompare",
            "--host",
            "a=http://localhost:11434",
            "--host",
            "b=http://127.0.0.1:11434",
            "--models",
            "m",
            "--quality-suite",
            suite,
            "--no-save",
            "--temperature",
            "0",
        ],
    )
    assert seen.get("temperature") == 0.0


def test_sweep_rejects_temperature_outside_zero_to_two():
    from click.testing import CliRunner

    from atomics.cli import cli

    result = CliRunner().invoke(cli, ["sweep", "--models", "a", "--temperature", "2.5"])
    assert result.exit_code == 2
    assert "not in the range" in result.output
