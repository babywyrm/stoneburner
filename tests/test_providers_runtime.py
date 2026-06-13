"""Runtime-path tests for provider generate/health check methods via DI."""

from __future__ import annotations

import pytest

from atomics.providers.base import ProviderResponse


class _FakeClaudeResp:
    class _Text:
        def __init__(self, text: str):
            self.text = text

    class _Usage:
        def __init__(self, input_tokens: int, output_tokens: int):
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

    def __init__(self):
        self.content = [self._Text("hello")]
        self.usage = self._Usage(input_tokens=10, output_tokens=20)

    def model_dump(self):
        return {"ok": True}


@pytest.mark.asyncio
async def test_claude_generate_and_health():
    class FakeMessages:
        async def create(self, **_kwargs):
            return _FakeClaudeResp()

    from atomics.providers.claude import ClaudeProvider

    fake_client = type("FakeClient", (), {"messages": FakeMessages()})()
    provider = ClaudeProvider(api_key="fake", client=fake_client)
    resp = await provider.generate("hi", model="claude-sonnet-4-6")
    assert isinstance(resp, ProviderResponse)
    assert resp.text == "hello"
    assert resp.total_tokens == 30
    assert resp.model == "claude-sonnet-4-6"
    assert await provider.health_check() is True


class _FakeClaudeCacheResp:
    class _Text:
        def __init__(self, text: str):
            self.text = text
            self.type = "text"

    class _Usage:
        def __init__(self):
            self.input_tokens = 10
            self.output_tokens = 20
            self.cache_read_input_tokens = 2000
            self.cache_creation_input_tokens = 400

    def __init__(self):
        self.content = [self._Text("cached")]
        self.usage = self._Usage()

    def model_dump(self):
        return {"ok": True}


@pytest.mark.asyncio
async def test_claude_generate_captures_cache_tokens():
    class FakeMessages:
        async def create(self, **_kwargs):
            return _FakeClaudeCacheResp()

    from atomics.providers.claude import ClaudeProvider, _estimate_cost

    fake_client = type("FakeClient", (), {"messages": FakeMessages()})()
    provider = ClaudeProvider(api_key="fake", client=fake_client)
    resp = await provider.generate("hi", model="claude-sonnet-4-20250514")

    assert resp.cache_read_tokens == 2000
    assert resp.cache_write_tokens == 400
    # Cost must reflect the cached-token discount/premium.
    expected = round(_estimate_cost("claude-sonnet-4-20250514", 10, 20, 2000, 400), 6)
    assert resp.estimated_cost_usd == expected


@pytest.mark.asyncio
async def test_claude_generate_without_cache_fields_defaults_zero():
    """A usage object lacking cache fields must not crash and reports 0."""
    class FakeMessages:
        async def create(self, **_kwargs):
            return _FakeClaudeResp()

    from atomics.providers.claude import ClaudeProvider

    fake_client = type("FakeClient", (), {"messages": FakeMessages()})()
    provider = ClaudeProvider(api_key="fake", client=fake_client)
    resp = await provider.generate("hi", model="claude-sonnet-4-6")
    assert resp.cache_read_tokens == 0
    assert resp.cache_write_tokens == 0


@pytest.mark.asyncio
async def test_claude_generate_reports_wall_clock_basis():
    class FakeMessages:
        async def create(self, **_kwargs):
            return _FakeClaudeResp()

    from atomics.providers.claude import ClaudeProvider

    fake_client = type("FakeClient", (), {"messages": FakeMessages()})()
    provider = ClaudeProvider(api_key="fake", client=fake_client)
    resp = await provider.generate("hi", model="claude-sonnet-4-6")
    assert resp.tps_basis == "wall_clock"


@pytest.mark.asyncio
async def test_claude_health_failure():
    class FakeMessages:
        async def create(self, **_kwargs):
            raise RuntimeError("boom")

    from atomics.providers.claude import ClaudeProvider

    fake_client = type("FakeClient", (), {"messages": FakeMessages()})()
    provider = ClaudeProvider(api_key="fake", client=fake_client)
    assert await provider.health_check() is False


@pytest.mark.asyncio
async def test_bedrock_generate_and_health():
    class FakeBedrockClient:
        def converse(self, **_kwargs):
            return {
                "output": {"message": {"content": [{"text": "ok"}]}},
                "usage": {"inputTokens": 4, "outputTokens": 6},
            }

    from atomics.providers.bedrock import BedrockProvider

    provider = BedrockProvider(region="us-east-1", client=FakeBedrockClient())
    resp = await provider.generate("hello")
    assert resp.text == "ok"
    assert resp.total_tokens == 10
    assert await provider.health_check() is True


class _FakeOpenAIChoice:
    def __init__(self, text: str):
        self.message = type("Msg", (), {"content": text})()


class _FakeOpenAIUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeOpenAIResp:
    def __init__(self):
        self.choices = [_FakeOpenAIChoice("world")]
        self.usage = _FakeOpenAIUsage(prompt_tokens=8, completion_tokens=12)

    def model_dump(self):
        return {"ok": True}


@pytest.mark.asyncio
async def test_openai_generate_and_health():
    class FakeCompletions:
        async def create(self, **_kwargs):
            return _FakeOpenAIResp()

    fake_client = type(
        "FakeClient", (), {"chat": type("Chat", (), {"completions": FakeCompletions()})()}
    )()

    from atomics.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key="fake", client=fake_client)
    resp = await provider.generate("hello", model="gpt-4o")
    assert isinstance(resp, ProviderResponse)
    assert resp.text == "world"
    assert resp.total_tokens == 20
    assert resp.model == "gpt-4o"
    assert await provider.health_check() is True


@pytest.mark.asyncio
async def test_openai_health_failure():
    class BadCompletions:
        async def create(self, **_kwargs):
            raise RuntimeError("boom")

    fake_client = type(
        "Bad", (), {"chat": type("Chat", (), {"completions": BadCompletions()})()}
    )()

    from atomics.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key="fake", client=fake_client)
    assert await provider.health_check() is False


@pytest.mark.asyncio
async def test_openai_health_passes_on_empty_reasoning_response():
    """A reasoning model can burn the whole budget and return empty visible text
    (finish_reason='length'); a token-consuming round-trip still means healthy."""
    class _EmptyChoice:
        def __init__(self):
            self.message = type("Msg", (), {"content": ""})()

    class _Resp:
        def __init__(self):
            self.choices = [_EmptyChoice()]
            # 8 prompt + 64 reasoning tokens consumed, but no visible content.
            self.usage = _FakeOpenAIUsage(prompt_tokens=8, completion_tokens=64)

        def model_dump(self):
            return {"ok": True}

    class FakeCompletions:
        async def create(self, **_kwargs):
            return _Resp()

    fake_client = type(
        "FakeClient", (), {"chat": type("Chat", (), {"completions": FakeCompletions()})()}
    )()

    from atomics.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key="fake", default_model="o4-mini", client=fake_client)
    resp = await provider.generate("hi", model="o4-mini", max_tokens=8)
    assert resp.text == ""
    assert resp.total_tokens == 72
    assert await provider.health_check() is True
