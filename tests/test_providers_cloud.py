"""Tests for Groq, Together, and Gemini providers (no live API calls)."""

from __future__ import annotations

import httpx
import pytest

from atomics.providers.base import BaseProvider

# ── Fake HTTP response helper ────────────────────────────────────────────────


def _fake_completion_response(
    text: str = "OK",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    model: str = "test-model",
    status_code: int = 200,
) -> httpx.Response:
    body = {
        "choices": [{"message": {"content": text}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "model": model,
    }
    request = httpx.Request("POST", "https://fake.api/v1/chat/completions")
    return httpx.Response(status_code, json=body, request=request)


class FakeClient(httpx.AsyncClient):
    def __init__(self, response: httpx.Response | None = None, *, fail: bool = False):
        super().__init__()
        self._fail = fail
        self._response = response or _fake_completion_response()
        self.post_calls: list[dict] = []
        self.get_calls: list[str] = []

    async def post(self, url, **kwargs):
        self.post_calls.append({"url": url, **kwargs})
        if self._fail:
            request = httpx.Request("POST", url)
            return httpx.Response(500, request=request)
        return self._response

    async def get(self, url, **kwargs):
        self.get_calls.append(url)
        if self._fail:
            request = httpx.Request("GET", url)
            return httpx.Response(500, request=request)
        return self._response


# ── Groq tests ───────────────────────────────────────────────────────────────


class TestGroqProvider:
    def _make(self, client: FakeClient | None = None):
        from atomics.providers.groq import GroqProvider

        return GroqProvider(
            api_key="fake-groq-key",
            client=client or FakeClient(),
        )

    def test_implements_base(self):
        prov = self._make()
        assert isinstance(prov, BaseProvider)
        assert prov.name == "groq"

    def test_default_model(self):
        prov = self._make()
        assert prov.default_model == "llama-3.3-70b-versatile"

    @pytest.mark.asyncio
    async def test_generate_parses_response(self):
        client = FakeClient(_fake_completion_response("hello from groq", 20, 10))
        prov = self._make(client)
        resp = await prov.generate("test")
        assert resp.text == "hello from groq"
        assert resp.input_tokens == 20
        assert resp.output_tokens == 10
        assert resp.total_tokens == 30
        assert len(client.post_calls) == 1
        assert "groq.com" in client.post_calls[0]["url"]

    @pytest.mark.asyncio
    async def test_generate_sends_auth_header(self):
        client = FakeClient()
        prov = self._make(client)
        await prov.generate("test")
        headers = client.post_calls[0]["headers"]
        assert "Bearer fake-groq-key" in headers.get("Authorization", "")

    @pytest.mark.asyncio
    async def test_generate_with_temperature(self):
        client = FakeClient()
        prov = self._make(client)
        await prov.generate("test", temperature=0.0)
        body = client.post_calls[0]["json"]
        assert body["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_health_check_true(self):
        prov = self._make()
        assert await prov.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_false_on_error(self):
        client = FakeClient(fail=True)
        prov = self._make(client)
        assert await prov.health_check() is False

    def test_cost_estimation(self):
        from atomics.providers.groq import _estimate_cost

        cost = _estimate_cost("llama-3.1-8b-instant", 1000, 500)
        expected = (1000 * 0.05 + 500 * 0.08) / 1_000_000
        assert abs(cost - expected) < 0.000001

    def test_cost_estimation_unknown_model(self):
        from atomics.providers.groq import _estimate_cost

        cost = _estimate_cost("unknown-model", 1000, 500)
        assert cost > 0


# ── Together tests ───────────────────────────────────────────────────────────


class TestTogetherProvider:
    def _make(self, client: FakeClient | None = None):
        from atomics.providers.together import TogetherProvider

        return TogetherProvider(
            api_key="fake-together-key",
            client=client or FakeClient(),
        )

    def test_implements_base(self):
        prov = self._make()
        assert isinstance(prov, BaseProvider)
        assert prov.name == "together"

    def test_default_model(self):
        prov = self._make()
        assert "llama" in prov.default_model.lower() or "Llama" in prov.default_model

    @pytest.mark.asyncio
    async def test_generate_parses_response(self):
        client = FakeClient(_fake_completion_response("hello from together", 15, 8))
        prov = self._make(client)
        resp = await prov.generate("test")
        assert resp.text == "hello from together"
        assert resp.input_tokens == 15
        assert resp.output_tokens == 8
        assert "together.xyz" in client.post_calls[0]["url"]

    @pytest.mark.asyncio
    async def test_generate_sends_auth_header(self):
        client = FakeClient()
        prov = self._make(client)
        await prov.generate("test")
        headers = client.post_calls[0]["headers"]
        assert "Bearer fake-together-key" in headers.get("Authorization", "")

    @pytest.mark.asyncio
    async def test_health_check_true(self):
        prov = self._make()
        assert await prov.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_false_on_error(self):
        client = FakeClient(fail=True)
        prov = self._make(client)
        assert await prov.health_check() is False

    def test_cost_estimation(self):
        from atomics.providers.together import _estimate_cost

        cost = _estimate_cost("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", 1000, 500)
        expected = (1000 * 0.18 + 500 * 0.18) / 1_000_000
        assert abs(cost - expected) < 0.000001


# ── Gemini tests ─────────────────────────────────────────────────────────────


class TestGeminiProvider:
    def _make(self, client: FakeClient | None = None):
        from atomics.providers.gemini import GeminiProvider

        return GeminiProvider(
            api_key="fake-gemini-key",
            client=client or FakeClient(),
        )

    def test_implements_base(self):
        prov = self._make()
        assert isinstance(prov, BaseProvider)
        assert prov.name == "gemini"

    def test_default_model(self):
        prov = self._make()
        assert prov.default_model == "gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_generate_parses_response(self):
        client = FakeClient(_fake_completion_response("hello from gemini", 12, 6))
        prov = self._make(client)
        resp = await prov.generate("test")
        assert resp.text == "hello from gemini"
        assert resp.input_tokens == 12
        assert resp.output_tokens == 6
        assert "generativelanguage" in client.post_calls[0]["url"]

    @pytest.mark.asyncio
    async def test_generate_sends_auth_header(self):
        client = FakeClient()
        prov = self._make(client)
        await prov.generate("test")
        headers = client.post_calls[0]["headers"]
        assert "Bearer fake-gemini-key" in headers.get("Authorization", "")

    @pytest.mark.asyncio
    async def test_generate_with_custom_model(self):
        client = FakeClient()
        prov = self._make(client)
        await prov.generate("test", model="gemini-2.5-pro")
        body = client.post_calls[0]["json"]
        assert body["model"] == "gemini-2.5-pro"

    @pytest.mark.asyncio
    async def test_health_check_true(self):
        prov = self._make()
        assert await prov.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_false_on_error(self):
        client = FakeClient(fail=True)
        prov = self._make(client)
        assert await prov.health_check() is False

    def test_cost_estimation(self):
        from atomics.providers.gemini import _estimate_cost

        cost = _estimate_cost("gemini-2.5-flash", 1000, 500)
        expected = (1000 * 0.15 + 500 * 0.60) / 1_000_000
        assert abs(cost - expected) < 0.000001

    def test_cost_estimation_pro(self):
        from atomics.providers.gemini import _estimate_cost

        cost = _estimate_cost("gemini-2.5-pro", 1000, 500)
        expected = (1000 * 1.25 + 500 * 10.0) / 1_000_000
        assert abs(cost - expected) < 0.000001

    @pytest.mark.asyncio
    async def test_thinking_tokens_from_details(self):
        body = {
            "choices": [{"message": {"content": "answer"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "completion_tokens_details": {"reasoning_tokens": 8},
            },
        }
        request = httpx.Request("POST", "https://fake.api/v1/chat/completions")
        client = FakeClient(httpx.Response(200, json=body, request=request))
        prov = self._make(client)
        resp = await prov.generate("test")
        assert resp.thinking_tokens == 8


# ── CLI integration tests ────────────────────────────────────────────────────


def test_cli_provider_test_groq_missing_key():
    from click.testing import CliRunner

    from atomics.cli import cli

    runner = CliRunner(env={"GROQ_API_KEY": ""})
    result = runner.invoke(cli, ["provider-test", "--provider", "groq"])
    assert result.exit_code != 0
    assert "GROQ_API_KEY" in result.output


def test_cli_provider_test_together_missing_key():
    from click.testing import CliRunner

    from atomics.cli import cli

    runner = CliRunner(env={"TOGETHER_API_KEY": ""})
    result = runner.invoke(cli, ["provider-test", "--provider", "together"])
    assert result.exit_code != 0
    assert "TOGETHER_API_KEY" in result.output


def test_cli_provider_test_gemini_missing_key():
    from click.testing import CliRunner

    from atomics.cli import cli

    runner = CliRunner(env={"GEMINI_API_KEY": ""})
    result = runner.invoke(cli, ["provider-test", "--provider", "gemini"])
    assert result.exit_code != 0
    assert "GEMINI_API_KEY" in result.output
