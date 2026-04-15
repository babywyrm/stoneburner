"""Unit tests for OpenAIProvider with injected clients (no real API calls)."""

from __future__ import annotations

import builtins
import sys

import pytest

from atomics.providers.openai import OpenAIProvider, _estimate_cost


class FakeChoice:
    def __init__(self, text: str) -> None:
        self.message = type("Msg", (), {"content": text})()


class FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeCompletionResponse:
    def __init__(
        self,
        text: str = "hello from openai",
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
    ) -> None:
        self.choices = [FakeChoice(text)]
        self.usage = FakeUsage(prompt_tokens, completion_tokens)

    def model_dump(self):
        return {"ok": True}


class FakeChatCompletions:
    def __init__(self) -> None:
        self.create_calls: list[dict] = []

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return FakeCompletionResponse()


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": FakeChatCompletions()})()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_generate_parses_response():
    client = FakeOpenAIClient()
    provider = OpenAIProvider(api_key="fake", client=client)
    resp = await provider.generate("ping", system="Be concise", max_tokens=256)

    assert resp.text == "hello from openai"
    assert resp.input_tokens == 100
    assert resp.output_tokens == 50
    assert resp.total_tokens == 150
    assert resp.model == "gpt-4o"
    assert client.chat.completions.create_calls
    call = client.chat.completions.create_calls[0]
    assert call["model"] == "gpt-4o"
    assert any(m["role"] == "system" for m in call["messages"])
    assert any(m["role"] == "user" for m in call["messages"])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_generate_with_custom_model():
    client = FakeOpenAIClient()
    provider = OpenAIProvider(api_key="fake", client=client, default_model="o3")
    resp = await provider.generate("test", model="gpt-4o-mini")
    assert resp.model == "gpt-4o-mini"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_generate_default_system_prompt():
    client = FakeOpenAIClient()
    provider = OpenAIProvider(api_key="fake", client=client)
    await provider.generate("test")
    call = client.chat.completions.create_calls[0]
    sys_msg = next(m for m in call["messages"] if m["role"] == "system")
    assert "helpful" in sys_msg["content"].lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_health_check_true():
    client = FakeOpenAIClient()
    provider = OpenAIProvider(api_key="fake", client=client)
    assert await provider.health_check() is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_health_check_false_on_error():
    class BadCompletions:
        async def create(self, **_kwargs):
            raise RuntimeError("API unavailable")

    bad_client = type("Bad", (), {"chat": type("C", (), {"completions": BadCompletions()})()})()
    provider = OpenAIProvider(api_key="fake", client=bad_client)
    assert await provider.health_check() is False


@pytest.mark.unit
def test_openai_name():
    client = FakeOpenAIClient()
    provider = OpenAIProvider(api_key="fake", client=client)
    assert provider.name == "openai"


@pytest.mark.unit
def test_openai_requires_sdk_when_no_client(monkeypatch):
    monkeypatch.delitem(sys.modules, "openai", raising=False)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="openai is required"):
        OpenAIProvider(api_key="fake", client=None)


@pytest.mark.unit
def test_openai_cost_estimation():
    cost = _estimate_cost("gpt-4o", 1000, 500)
    expected = (1000 * 2.50 + 500 * 10.0) / 1_000_000
    assert abs(cost - expected) < 0.000001


@pytest.mark.unit
def test_openai_cost_estimation_mini():
    cost = _estimate_cost("gpt-4o-mini", 1000, 500)
    expected = (1000 * 0.15 + 500 * 0.60) / 1_000_000
    assert abs(cost - expected) < 0.000001


@pytest.mark.unit
def test_openai_cost_estimation_unknown_model():
    cost = _estimate_cost("unknown-model", 1000, 500)
    expected = (1000 * 2.50 + 500 * 10.0) / 1_000_000
    assert abs(cost - expected) < 0.000001


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_generate_no_usage():
    """Handle responses where usage is None."""

    class NoUsageResponse:
        choices = [FakeChoice("hi")]
        usage = None

        def model_dump(self):
            return {}

    class NoUsageCompletions:
        async def create(self, **_kwargs):
            return NoUsageResponse()

    client = type("C", (), {"chat": type("CH", (), {"completions": NoUsageCompletions()})()})()
    provider = OpenAIProvider(api_key="fake", client=client)
    resp = await provider.generate("test")
    assert resp.input_tokens == 0
    assert resp.output_tokens == 0
    assert resp.total_tokens == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_generate_empty_choices():
    """Handle responses with no choices."""

    class EmptyResponse:
        choices = []
        usage = FakeUsage(10, 5)

        def model_dump(self):
            return {}

    class EmptyCompletions:
        async def create(self, **_kwargs):
            return EmptyResponse()

    client = type("C", (), {"chat": type("CH", (), {"completions": EmptyCompletions()})()})()
    provider = OpenAIProvider(api_key="fake", client=client)
    resp = await provider.generate("test")
    assert resp.text == ""
    assert resp.total_tokens == 15


# ── Responses API path (OAuth) ──────────────────────────


class FakeOutputText:
    def __init__(self, text: str) -> None:
        self.type = "output_text"
        self.text = text


class FakeOutputMessage:
    def __init__(self, text: str) -> None:
        self.type = "message"
        self.content = [FakeOutputText(text)]


class FakeResponsesUsage:
    def __init__(self, input_tokens: int = 20, output_tokens: int = 30) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeResponsesResponse:
    def __init__(self, text: str = "hello from responses api") -> None:
        self.output_text = text
        self.output = [FakeOutputMessage(text)]
        self.usage = FakeResponsesUsage()

    def model_dump(self):
        return {"ok": True}


class FakeResponses:
    def __init__(self) -> None:
        self.create_calls: list[dict] = []

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return FakeResponsesResponse()


class FakeOAuthClient:
    """Client that has both chat.completions and responses."""

    def __init__(self) -> None:
        self.responses = FakeResponses()
        self.chat = type("Chat", (), {"completions": FakeChatCompletions()})()
        self.api_key = "oauth-managed"


class FakeAuth:
    """Minimal auth strategy for testing."""

    async def get_headers(self):
        return {"Authorization": "Bearer fake-oauth-token"}

    async def validate(self):
        return True

    @property
    def description(self):
        return "Fake OAuth"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_responses_api_used_with_auth():
    client = FakeOAuthClient()
    provider = OpenAIProvider(client=client, auth=FakeAuth())
    resp = await provider.generate("ping", system="Be concise", max_tokens=64)

    assert resp.text == "hello from responses api"
    assert resp.input_tokens == 20
    assert resp.output_tokens == 30
    assert resp.total_tokens == 50
    assert client.responses.create_calls
    assert not client.chat.completions.create_calls


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_responses_api_passes_model():
    client = FakeOAuthClient()
    provider = OpenAIProvider(client=client, auth=FakeAuth(), default_model="o4-mini")
    await provider.generate("test")
    call = client.responses.create_calls[0]
    assert call["model"] == "o4-mini"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_responses_api_sets_token():
    client = FakeOAuthClient()
    provider = OpenAIProvider(client=client, auth=FakeAuth())
    await provider.generate("test")
    assert client.api_key == "fake-oauth-token"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_completions_api_used_without_auth():
    client = FakeOAuthClient()
    provider = OpenAIProvider(api_key="sk-test", client=client)
    await provider.generate("test")
    assert client.chat.completions.create_calls
    assert not client.responses.create_calls


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_responses_api_health_check():
    client = FakeOAuthClient()
    provider = OpenAIProvider(client=client, auth=FakeAuth())
    assert await provider.health_check() is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_responses_api_fallback_text_extraction():
    """Test text extraction from output when output_text is empty."""

    class NoOutputTextResp:
        output_text = ""
        output = [FakeOutputMessage("fallback text")]
        usage = FakeResponsesUsage()

        def model_dump(self):
            return {}

    class FallbackResponses:
        async def create(self, **kwargs):
            return NoOutputTextResp()

    client = type("C", (), {
        "responses": FallbackResponses(),
        "chat": type("CH", (), {"completions": FakeChatCompletions()})(),
        "api_key": "x",
    })()
    provider = OpenAIProvider(client=client, auth=FakeAuth())
    resp = await provider.generate("test")
    assert resp.text == "fallback text"
