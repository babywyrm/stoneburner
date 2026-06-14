"""Temperature plumbing across providers + deterministic judge scoring.

Verifies that an explicit temperature is forwarded to each backend where it is
valid, and is correctly withheld for the two cases that reject/forbid it:
OpenAI reasoning models and Claude with extended thinking enabled.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from atomics.providers.base import ProviderResponse


# ── Ollama ──────────────────────────────────────────────────────────────────
def _ollama_response():
    r = MagicMock()
    r.status_code = 200
    r.raise_for_status = MagicMock()
    r.json.return_value = {
        "response": "ok", "eval_count": 5, "prompt_eval_count": 3,
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
    assert msgs.kwargs["temperature"] == 0.0


@pytest.mark.asyncio
async def test_claude_omits_temperature_with_thinking():
    provider, msgs = _claude_provider()
    await provider.generate(
        "hi", model="claude-sonnet-4-6", thinking=True, temperature=0.0,
    )
    # Anthropic requires temperature=1 with extended thinking, so we must not
    # send our explicit value.
    assert "temperature" not in msgs.kwargs
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


# ── Judge passes temperature=0.0 ────────────────────────────────────────────
class _CapturingJudgeProvider:
    """Records the temperature the judge requests; returns a parseable rubric."""
    name = "fake-judge"

    def __init__(self):
        self.temperature = "unset"

    async def generate(self, prompt, *, system="", model=None, max_tokens=1024,
                        thinking=None, thinking_budget=None, temperature=None):
        self.temperature = temperature
        return ProviderResponse(
            text="ACCURACY: 4\nCOMPLETENESS: 3\nFORMAT: 3\nRATIONALE: solid.",
            input_tokens=1, output_tokens=1, total_tokens=2,
            model="fake-judge", latency_ms=1.0, estimated_cost_usd=0.0,
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

    async def _gen(prompt, *, system="", model=None, max_tokens=1024,
                   thinking=None, thinking_budget=None, temperature=None):
        judge.temperature = temperature
        return ProviderResponse(
            text=judge_text, input_tokens=1, output_tokens=1, total_tokens=2,
            model="fake-judge", latency_ms=1.0, estimated_cost_usd=0.0,
        )

    judge.generate = _gen
    result = await score_resistance(
        "p", "r", attack_goal="g", resistance_criteria=["c"], judge_provider=judge,
    )
    assert judge.temperature == 0.0
    assert result.label == "resisted"
