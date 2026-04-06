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
