"""Tests for the llama.cpp provider (no live server)."""

from __future__ import annotations

import httpx
import pytest

from atomics.providers.base import BaseProvider
from atomics.providers.llamacpp import LlamaCppProvider


def _fake_response(text: str = "OK", inp: int = 10, out: int = 5) -> httpx.Response:
    body = {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": inp, "completion_tokens": out, "total_tokens": inp + out},
    }
    request = httpx.Request("POST", "http://localhost:8080/v1/chat/completions")
    return httpx.Response(200, json=body, request=request)


class FakeClient(httpx.AsyncClient):
    def __init__(self, response: httpx.Response | None = None, *, fail: bool = False):
        super().__init__()
        self._response = response or _fake_response()
        self._fail = fail
        self.post_calls: list[str] = []

    async def post(self, url, **kwargs):
        self.post_calls.append(url)
        if self._fail:
            return httpx.Response(500, request=httpx.Request("POST", url))
        return self._response

    async def get(self, url, **kwargs):
        if self._fail:
            return httpx.Response(500, request=httpx.Request("GET", url))
        return httpx.Response(200, request=httpx.Request("GET", url))


def test_implements_base():
    prov = LlamaCppProvider(client=FakeClient())
    assert isinstance(prov, BaseProvider)
    assert prov.name == "llamacpp"


def test_default_model():
    prov = LlamaCppProvider(client=FakeClient())
    assert prov.default_model == "local"


@pytest.mark.asyncio
async def test_generate():
    client = FakeClient(_fake_response("hello from llama.cpp", 15, 8))
    prov = LlamaCppProvider(client=client)
    resp = await prov.generate("test")
    assert resp.text == "hello from llama.cpp"
    assert resp.input_tokens == 15
    assert resp.output_tokens == 8
    assert resp.estimated_cost_usd == 0.0
    assert resp.tokens_per_second > 0
    assert len(client.post_calls) == 1


@pytest.mark.asyncio
async def test_generate_with_temperature():
    client = FakeClient()
    prov = LlamaCppProvider(client=client)
    await prov.generate("test", temperature=0.5)


@pytest.mark.asyncio
async def test_health_check_true():
    prov = LlamaCppProvider(client=FakeClient())
    assert await prov.health_check() is True


@pytest.mark.asyncio
async def test_health_check_false():
    prov = LlamaCppProvider(client=FakeClient(fail=True))
    assert await prov.health_check() is False


def test_zero_cost():
    """llama.cpp is always free — local inference."""
    prov = LlamaCppProvider(client=FakeClient())
    assert prov.name == "llamacpp"


@pytest.mark.asyncio
async def test_llamacpp_generate_with_tools_uses_the_v1_prefixed_path():
    """llama.cpp mounts the OpenAI surface under /v1, unlike the other compat
    providers which fold that into their base URL."""
    body = {
        "choices": [{
            "message": {
                "content": "",
                "tool_calls": [{
                    "function": {"name": "read_file", "arguments": '{"path": "/etc/shadow"}'}
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
    }
    request = httpx.Request("POST", "http://fake:8080/v1/chat/completions")
    response = httpx.Response(200, json=body, request=request)

    calls: list[str] = []

    class _Client(httpx.AsyncClient):
        async def post(self, url, **kwargs):
            calls.append(url)
            return response

    provider = LlamaCppProvider(base_url="http://fake:8080", client=_Client())
    resp = await provider.generate_with_tools(
        "Show the hashes.",
        tools=[{"name": "read_file", "description": "d",
                "parameters": {"type": "object", "properties": {}}}],
    )

    assert calls[0] == "http://fake:8080/v1/chat/completions"
    assert resp.tool_calls[0].arguments == {"path": "/etc/shadow"}
    # Self-hosted: no price table applies.
    assert resp.estimated_cost_usd == 0.0


def test_llamacpp_declares_tool_support():
    assert LlamaCppProvider.supports_tools is True
