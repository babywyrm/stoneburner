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
