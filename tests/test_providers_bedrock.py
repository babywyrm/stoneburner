"""Unit tests for BedrockProvider with injected clients (no AWS calls)."""

from __future__ import annotations

import builtins
import sys

import pytest

from atomics.providers.bedrock import BedrockProvider


class FakeBedrockClient:
    def __init__(self) -> None:
        self.converse_calls: list[dict] = []

    def converse(self, **kwargs):
        self.converse_calls.append(kwargs)
        return {
            "output": {"message": {"content": [{"text": "hello from bedrock"}]}},
            "usage": {"inputTokens": 100, "outputTokens": 50},
        }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bedrock_generate_parses_converse_response():
    client = FakeBedrockClient()
    provider = BedrockProvider(region="us-west-2", client=client)
    resp = await provider.generate("ping", system="Be concise", max_tokens=256)

    assert resp.text == "hello from bedrock"
    assert resp.input_tokens == 100
    assert resp.output_tokens == 50
    assert resp.total_tokens == 150
    assert "anthropic" in resp.model
    assert client.converse_calls
    assert client.converse_calls[0]["system"] == [{"text": "Be concise"}]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bedrock_health_check_true_when_text_returned():
    client = FakeBedrockClient()
    provider = BedrockProvider(client=client)
    assert await provider.health_check() is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bedrock_health_check_false_on_error():
    class BadClient:
        def converse(self, **_kwargs):
            raise RuntimeError("AWS unavailable")

    provider = BedrockProvider(client=BadClient())
    assert await provider.health_check() is False


@pytest.mark.unit
def test_bedrock_requires_boto3_when_no_client(monkeypatch):
    monkeypatch.delitem(sys.modules, "boto3", raising=False)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "boto3":
            raise ImportError("No module named 'boto3'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="boto3 is required"):
        BedrockProvider(region="us-east-1", client=None)
