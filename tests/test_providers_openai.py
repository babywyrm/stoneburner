"""Unit tests for OpenAIProvider with injected clients (no real API calls)."""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest

from atomics.eval import ProviderOutcome, ProviderOutcomeKind
from atomics.providers.base import ProviderResponse
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
async def test_openai_generate_prices_dated_gpt_5_6_model():
    client = FakeOpenAIClient()
    provider = OpenAIProvider(api_key="fake", client=client)

    response = await provider.generate(
        "ping",
        model="gpt-5.6-2026-06-01",
    )

    assert response.estimated_cost_usd == 0.002


@pytest.mark.unit
def test_provider_response_outcome_metadata_is_backward_compatible():
    response = ProviderResponse(
        text="legacy",
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        model="fake",
        latency_ms=1.0,
        estimated_cost_usd=0.0,
    )

    assert response.outcome is None
    assert response.finish_reason is None
    response.finish_reason = "later"
    assert response.finish_reason == "later"


@pytest.mark.unit
def test_provider_response_fills_finish_reason_from_outcome():
    response = ProviderResponse(
        text="partial",
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        model="fake",
        latency_ms=1.0,
        estimated_cost_usd=0.0,
        outcome=ProviderOutcome(
            ProviderOutcomeKind.TRUNCATED,
            finish_reason="length",
        ),
    )

    assert response.finish_reason == "length"


@pytest.mark.unit
def test_provider_response_rejects_conflicting_finish_reasons():
    with pytest.raises(ValueError, match="finish_reason"):
        ProviderResponse(
            text="partial",
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            model="fake",
            latency_ms=1.0,
            estimated_cost_usd=0.0,
            outcome=ProviderOutcome(
                ProviderOutcomeKind.TRUNCATED,
                finish_reason="length",
            ),
            finish_reason="stop",
        )


@pytest.mark.unit
@pytest.mark.parametrize("finish_reason", [None, "stop"])
def test_provider_response_rejects_later_conflicting_finish_reason(
    finish_reason: str | None,
):
    response = ProviderResponse(
        text="partial",
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        model="fake",
        latency_ms=1.0,
        estimated_cost_usd=0.0,
        outcome=ProviderOutcome(
            ProviderOutcomeKind.TRUNCATED,
            finish_reason="length",
        ),
    )

    with pytest.raises(ValueError, match="finish_reason"):
        response.finish_reason = finish_reason

    assert response.finish_reason == "length"


@pytest.mark.unit
def test_provider_response_rejects_later_conflicting_outcome_assignment():
    response = ProviderResponse(
        text="legacy",
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        model="fake",
        latency_ms=1.0,
        estimated_cost_usd=0.0,
        finish_reason="stop",
    )

    with pytest.raises(ValueError, match="finish_reason"):
        response.outcome = ProviderOutcome(
            ProviderOutcomeKind.TRUNCATED,
            finish_reason="length",
        )

    assert response.outcome is None


@pytest.mark.unit
def test_provider_response_rejects_conflicting_outcome_replacement():
    original = ProviderOutcome(
        ProviderOutcomeKind.TRUNCATED,
        finish_reason="length",
    )
    response = ProviderResponse(
        text="partial",
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        model="fake",
        latency_ms=1.0,
        estimated_cost_usd=0.0,
        outcome=original,
    )

    with pytest.raises(ValueError, match="finish_reason"):
        response.outcome = ProviderOutcome(
            ProviderOutcomeKind.COMPLETED,
            finish_reason="stop",
        )

    assert response.outcome is original


@pytest.mark.unit
def test_provider_response_allows_outcome_replacement_with_same_finish_reason():
    response = ProviderResponse(
        text="partial",
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        model="fake",
        latency_ms=1.0,
        estimated_cost_usd=0.0,
        outcome=ProviderOutcome(
            ProviderOutcomeKind.TRUNCATED,
            finish_reason="length",
        ),
    )
    replacement = ProviderOutcome(
        ProviderOutcomeKind.REFUSED,
        finish_reason="length",
    )

    response.outcome = replacement

    assert response.outcome is replacement
    assert response.finish_reason == "length"


def _chat_client(
    *,
    content: str | None,
    finish_reason: str,
    refusal: str | None = None,
    reasoning_content: str = "",
    completion_tokens: int = 5,
):
    message = SimpleNamespace(
        content=content,
        refusal=refusal,
        reasoning_content=reasoning_content,
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    response = SimpleNamespace(
        choices=[choice],
        usage=FakeUsage(10, completion_tokens),
    )

    class Completions:
        async def create(self, **_kwargs):
            return response

    return SimpleNamespace(
        chat=SimpleNamespace(completions=Completions()),
    )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("finish_reason", "expected_kind"),
    [
        ("stop", ProviderOutcomeKind.COMPLETED),
        ("length", ProviderOutcomeKind.TRUNCATED),
        ("content_filter", ProviderOutcomeKind.SAFETY_BLOCKED),
    ],
)
async def test_openai_chat_normalizes_finish_reason(
    finish_reason: str,
    expected_kind: ProviderOutcomeKind,
):
    provider = OpenAIProvider(
        api_key="fake",
        client=_chat_client(content="visible", finish_reason=finish_reason),
    )

    response = await provider.generate("test")

    assert response.finish_reason == finish_reason
    assert response.outcome is not None
    assert response.outcome.kind is expected_kind
    assert response.outcome.finish_reason == finish_reason


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_chat_preserves_refusal_without_content():
    provider = OpenAIProvider(
        api_key="fake",
        client=_chat_client(
            content=None,
            finish_reason="stop",
            refusal="I cannot help with that.",
        ),
    )

    response = await provider.generate("test")

    assert response.text == "I cannot help with that."
    assert response.outcome is not None
    assert response.outcome.kind is ProviderOutcomeKind.REFUSED
    assert response.outcome.finish_reason == "stop"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_chat_combines_visible_and_refusal_text():
    provider = OpenAIProvider(
        api_key="fake",
        client=_chat_client(
            content="Visible partial answer.",
            finish_reason="stop",
            refusal="I cannot continue.",
        ),
    )

    response = await provider.generate("test")

    assert response.text == "Visible partial answer.\nI cannot continue."
    assert response.outcome is not None
    assert response.outcome.kind is ProviderOutcomeKind.REFUSED


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_chat_token_consuming_reasoning_without_visible_text_is_empty():
    provider = OpenAIProvider(
        api_key="fake",
        client=_chat_client(
            content=None,
            finish_reason="stop",
            reasoning_content="hidden chain of thought",
            completion_tokens=64,
        ),
    )

    response = await provider.generate("test")

    assert response.text == ""
    assert response.total_tokens == 74
    assert response.outcome is not None
    assert response.outcome.kind is ProviderOutcomeKind.EMPTY


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
        self.status = "completed"

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


def _responses_client(response):
    class Responses:
        async def create(self, **_kwargs):
            return response

    return SimpleNamespace(
        responses=Responses(),
        chat=SimpleNamespace(completions=FakeChatCompletions()),
        api_key="oauth-managed",
    )


def _responses_response(
    *,
    status: str,
    text: str = "",
    output: list | None = None,
    incomplete_reason: str | None = None,
    error=None,
):
    return SimpleNamespace(
        status=status,
        output_text=text,
        output=output or [],
        usage=FakeResponsesUsage(),
        incomplete_details=(
            SimpleNamespace(reason=incomplete_reason)
            if incomplete_reason is not None
            else None
        ),
        error=error,
    )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "expected_kind"),
    [
        ("visible", ProviderOutcomeKind.COMPLETED),
        ("", ProviderOutcomeKind.EMPTY),
    ],
)
async def test_openai_responses_normalizes_completed_status(
    text: str,
    expected_kind: ProviderOutcomeKind,
):
    provider = OpenAIProvider(
        client=_responses_client(
            _responses_response(status="completed", text=text)
        ),
        auth=FakeAuth(),
    )

    response = await provider.generate("test")

    assert response.finish_reason == "completed"
    assert response.outcome is not None
    assert response.outcome.kind is expected_kind
    assert response.outcome.finish_reason == "completed"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_responses_normalizes_incomplete_status():
    provider = OpenAIProvider(
        client=_responses_client(
            _responses_response(
                status="incomplete",
                text="partial",
                incomplete_reason="max_output_tokens",
            )
        ),
        auth=FakeAuth(),
    )

    response = await provider.generate("test")

    assert response.finish_reason == "max_output_tokens"
    assert response.outcome is not None
    assert response.outcome.kind is ProviderOutcomeKind.TRUNCATED
    assert response.outcome.finish_reason == "max_output_tokens"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_responses_preserves_refusal_text():
    refusal = SimpleNamespace(
        type="refusal",
        refusal="I cannot provide that.",
    )
    message = SimpleNamespace(type="message", content=[refusal])
    provider = OpenAIProvider(
        client=_responses_client(
            _responses_response(status="completed", output=[message])
        ),
        auth=FakeAuth(),
    )

    response = await provider.generate("test")

    assert response.text == "I cannot provide that."
    assert response.outcome is not None
    assert response.outcome.kind is ProviderOutcomeKind.REFUSED


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_responses_combines_visible_and_refusal_text_without_duplication():
    refusal = SimpleNamespace(type="refusal", refusal="I cannot continue.")
    message = SimpleNamespace(
        type="message",
        content=[
            SimpleNamespace(type="output_text", text="Visible partial answer."),
            refusal,
        ],
    )
    provider = OpenAIProvider(
        client=_responses_client(
            _responses_response(
                status="completed",
                text="Visible partial answer.",
                output=[message],
            )
        ),
        auth=FakeAuth(),
    )

    response = await provider.generate("test")

    assert response.text == "Visible partial answer.\nI cannot continue."
    assert response.outcome is not None
    assert response.outcome.kind is ProviderOutcomeKind.REFUSED


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "error", "incomplete_reason", "expected_kind"),
    [
        (
            "failed",
            SimpleNamespace(
                code="content_policy_violation",
                message="Blocked by content policy.",
            ),
            None,
            ProviderOutcomeKind.SAFETY_BLOCKED,
        ),
        (
            "failed",
            SimpleNamespace(code="server_error", message="Internal failure."),
            None,
            ProviderOutcomeKind.PROVIDER_ERROR,
        ),
        (
            "incomplete",
            None,
            "max_output_tokens",
            ProviderOutcomeKind.TRUNCATED,
        ),
    ],
)
async def test_openai_responses_status_precedes_refusal(
    status: str,
    error,
    incomplete_reason: str | None,
    expected_kind: ProviderOutcomeKind,
):
    refusal = SimpleNamespace(type="refusal", refusal="I cannot continue.")
    message = SimpleNamespace(type="message", content=[refusal])
    provider = OpenAIProvider(
        client=_responses_client(
            _responses_response(
                status=status,
                output=[message],
                incomplete_reason=incomplete_reason,
                error=error,
            )
        ),
        auth=FakeAuth(),
    )

    response = await provider.generate("test")

    assert response.text == "I cannot continue."
    assert response.outcome is not None
    assert response.outcome.kind is expected_kind


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_kind"),
    [
        (
            SimpleNamespace(
                code="content_policy_violation",
                message="Blocked by content policy.",
            ),
            ProviderOutcomeKind.SAFETY_BLOCKED,
        ),
        (
            SimpleNamespace(code="server_error", message="Internal failure."),
            ProviderOutcomeKind.PROVIDER_ERROR,
        ),
        (
            SimpleNamespace(
                code="invalid_request_error",
                message="Safety settings were malformed.",
            ),
            ProviderOutcomeKind.PROVIDER_ERROR,
        ),
    ],
)
async def test_openai_responses_normalizes_failed_status(
    error,
    expected_kind: ProviderOutcomeKind,
):
    provider = OpenAIProvider(
        client=_responses_client(
            _responses_response(status="failed", error=error)
        ),
        auth=FakeAuth(),
    )

    response = await provider.generate("test")

    assert response.finish_reason == error.code
    assert response.outcome is not None
    assert response.outcome.kind is expected_kind
    assert response.outcome.finish_reason == error.code
    if expected_kind is ProviderOutcomeKind.SAFETY_BLOCKED:
        assert response.outcome.safety_reason == "content_policy_violation"


def _official_sdk_response(
    *,
    status: str,
    incomplete_reason: str | None = None,
    error: dict[str, str] | None = None,
):
    pytest.importorskip("openai", reason="optional 'openai' extra not installed")
    from openai.types.responses import Response

    payload = {
        "id": "resp_status_test",
        "created_at": 1.0,
        "model": "gpt-5.6-2026-06-01",
        "object": "response",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": status,
    }
    if incomplete_reason is not None:
        payload["incomplete_details"] = {"reason": incomplete_reason}
    if error is not None:
        payload["error"] = error
    return Response.model_validate(payload)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_official_incomplete_content_filter_is_safety_blocked():
    provider = OpenAIProvider(
        client=_responses_client(
            _official_sdk_response(
                status="incomplete",
                incomplete_reason="content_filter",
            )
        ),
        auth=FakeAuth(),
    )

    response = await provider.generate("test")

    assert response.finish_reason == "content_filter"
    assert response.outcome is not None
    assert response.outcome.kind is ProviderOutcomeKind.SAFETY_BLOCKED
    assert response.outcome.safety_reason == "content_filter"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_official_image_policy_failure_is_safety_blocked():
    provider = OpenAIProvider(
        client=_responses_client(
            _official_sdk_response(
                status="failed",
                error={
                    "code": "image_content_policy_violation",
                    "message": "Image request rejected.",
                },
            )
        ),
        auth=FakeAuth(),
    )

    response = await provider.generate("test")

    assert response.outcome is not None
    assert response.outcome.kind is ProviderOutcomeKind.SAFETY_BLOCKED
    assert response.outcome.safety_reason == "image_content_policy_violation"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["cancelled", "queued", "in_progress"])
async def test_openai_official_nonterminal_status_is_provider_error(status: str):
    provider = OpenAIProvider(
        client=_responses_client(_official_sdk_response(status=status)),
        auth=FakeAuth(),
    )

    response = await provider.generate("test")

    assert response.finish_reason == status
    assert response.outcome is not None
    assert response.outcome.kind is ProviderOutcomeKind.PROVIDER_ERROR
    assert response.outcome.error_message == status


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_responses_accepts_official_sdk_response_model():
    pytest.importorskip("openai", reason="optional 'openai' extra not installed")
    from openai.types.responses import Response

    sdk_response = Response.model_validate(
        {
            "id": "resp_test",
            "created_at": 1.0,
            "model": "gpt-5.6-2026-06-01",
            "object": "response",
            "output": [
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "sdk visible",
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
            "usage": {
                "input_tokens": 4,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 3,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 7,
            },
        }
    )
    provider = OpenAIProvider(
        client=_responses_client(sdk_response),
        auth=FakeAuth(),
    )

    response = await provider.generate("test")

    assert response.text == "sdk visible"
    assert response.total_tokens == 7
    assert response.outcome is not None
    assert response.outcome.kind is ProviderOutcomeKind.COMPLETED


@pytest.mark.unit
def test_openai_timeout_configured():
    """OpenAI provider should set a 60s request timeout and 10s connect timeout."""
    pytest.importorskip("openai", reason="optional 'openai' extra not installed")
    provider = OpenAIProvider(api_key="sk-test-key")
    timeout = provider._client.timeout
    assert timeout.read == 60.0
    assert timeout.connect == 10.0


@pytest.mark.unit
def test_openai_max_retries_configured():
    """OpenAI provider should cap retries at 2."""
    pytest.importorskip("openai", reason="optional 'openai' extra not installed")
    provider = OpenAIProvider(api_key="sk-test-key")
    assert provider._client.max_retries == 2
