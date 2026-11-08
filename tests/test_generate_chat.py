"""`generate_chat`: earlier turns go to the model as messages, not pasted text."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.console import Console

from atomics.eval.budget import EvalBudget, GuardedProvider
from atomics.providers import trace
from atomics.providers.base import BaseProvider, ChatMessage, ProviderResponse
from atomics.providers.ollama import OllamaProvider

HISTORY: list[ChatMessage] = [
    {"role": "user", "content": "My name is Alex."},
    {"role": "assistant", "content": "Hi Alex."},
    {"role": "user", "content": "What is my name?"},
]


class _Recorder(BaseProvider):
    def __init__(self) -> None:
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return "rec"

    async def generate(self, prompt: str, **kwargs: object) -> ProviderResponse:  # type: ignore[override]
        self.calls.append({"prompt": prompt, **kwargs})
        return ProviderResponse(
            text="Alex",
            input_tokens=10,
            output_tokens=2,
            total_tokens=12,
            model="m",
            latency_ms=1.0,
            estimated_cost_usd=0.25,
        )

    async def health_check(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_default_pastes_history_exactly_as_before() -> None:
    rec = _Recorder()
    await rec.generate_chat(HISTORY, system="Be brief.", max_tokens=64)
    assert rec.calls[0]["system"] == ""
    assert rec.calls[0]["prompt"] == (
        "[System]: Be brief.\n\n[User]: My name is Alex.\n\n"
        "[Assistant]: Hi Alex.\n\n[User]: What is my name?"
    )
    await rec.generate_chat(HISTORY[:1], system="Be brief.")
    assert rec.calls[1]["system"] == "Be brief."
    assert rec.calls[1]["prompt"] == "My name is Alex."


@pytest.mark.asyncio
async def test_ollama_sends_turns_as_chat_messages() -> None:
    reply = MagicMock()
    reply.raise_for_status = MagicMock()
    reply.json.return_value = {
        "message": {"content": "Alex"},
        "eval_count": 2,
        "prompt_eval_count": 30,
        "eval_duration": 1,
        "done_reason": "stop",
    }
    client = AsyncMock()
    client.post = AsyncMock(return_value=reply)
    provider = OllamaProvider(default_model="m", client=client)

    response = await provider.generate_chat(
        HISTORY, system="Be brief.", max_tokens=64, temperature=0.0, thinking=False
    )

    url = client.post.call_args.args[0]
    body = client.post.call_args.kwargs["json"]
    assert url.endswith("/api/chat")
    assert body["messages"] == [{"role": "system", "content": "Be brief."}, *HISTORY]
    assert "tools" not in body
    assert body["options"]["num_predict"] == 64
    assert body["options"]["temperature"] == 0.0
    assert response.text == "Alex" and response.input_tokens == 30
    assert response.outcome is None


@pytest.mark.asyncio
async def test_claude_sends_turns_as_chat_messages() -> None:
    from atomics.providers.claude import ClaudeProvider

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Alex"
    usage = MagicMock()
    usage.input_tokens = 25
    usage.output_tokens = 2
    usage.thinking_tokens = 0
    mock_response = MagicMock()
    mock_response.content = [text_block]
    mock_response.usage = usage
    mock_response.model_dump.return_value = {}
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    provider = ClaudeProvider(api_key="test", client=mock_client)
    resp = await provider.generate_chat(HISTORY, system="Be brief.", max_tokens=64, temperature=0.0)

    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs["system"] == "Be brief."
    assert kwargs["messages"] == HISTORY
    assert kwargs["extra_body"] == {"temperature": 0.0}
    assert resp.text == "Alex"


@pytest.mark.asyncio
async def test_openai_completions_sends_turns_as_chat_messages() -> None:
    from atomics.providers.openai import OpenAIProvider
    from tests.test_providers_openai import FakeOpenAIClient

    client = FakeOpenAIClient()
    provider = OpenAIProvider(api_key="fake", client=client)
    await provider.generate_chat(HISTORY, system="Be brief.", max_tokens=64, temperature=0.0)

    call = client.chat.completions.create_calls[0]
    assert call["messages"] == [{"role": "system", "content": "Be brief."}, *HISTORY]
    assert call["temperature"] == 0.0


@pytest.mark.asyncio
async def test_openai_responses_sends_turns_as_input_items() -> None:
    from atomics.providers.openai import OpenAIProvider
    from tests.test_providers_openai import FakeAuth, FakeOAuthClient

    client = FakeOAuthClient()
    provider = OpenAIProvider(client=client, auth=FakeAuth())
    await provider.generate_chat(HISTORY, system="Be brief.", max_tokens=64)

    call = client.responses.create_calls[0]
    assert call["input"] == [{"role": "system", "content": "Be brief."}, *HISTORY]
    assert call["instructions"] == "Be brief."


@pytest.mark.asyncio
async def test_bedrock_sends_turns_as_converse_messages() -> None:
    from atomics.providers.bedrock import BedrockProvider

    mock_client = MagicMock()
    mock_client.converse.return_value = {
        "output": {"message": {"content": [{"text": "Alex"}]}},
        "usage": {"inputTokens": 20, "outputTokens": 2},
    }
    provider = BedrockProvider(client=mock_client)
    resp = await provider.generate_chat(HISTORY, system="Be brief.", max_tokens=64, temperature=0.0)

    kwargs = mock_client.converse.call_args.kwargs
    assert kwargs["system"] == [{"text": "Be brief."}]
    assert kwargs["messages"] == [
        {"role": "user", "content": [{"text": "My name is Alex."}]},
        {"role": "assistant", "content": [{"text": "Hi Alex."}]},
        {"role": "user", "content": [{"text": "What is my name?"}]},
    ]
    assert kwargs["inferenceConfig"]["temperature"] == 0.0
    assert resp.text == "Alex"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cls_path", "init_kwargs", "reply_json"),
    [
        (
            "atomics.providers.vllm:VllmProvider",
            {"default_model": "qwen2.5:3b"},
            {"choices": [{"message": {"content": "Alex"}}], "usage": {"prompt_tokens": 10}},
        ),
        (
            "atomics.providers.llamacpp:LlamaCppProvider",
            {"default_model": "local"},
            {"choices": [{"message": {"content": "Alex"}}], "usage": {"prompt_tokens": 10}},
        ),
        (
            "atomics.providers.gemini:GeminiProvider",
            {"api_key": "fake"},
            {"choices": [{"message": {"content": "Alex"}}], "usage": {"prompt_tokens": 10}},
        ),
        (
            "atomics.providers.groq:GroqProvider",
            {"api_key": "fake"},
            {"choices": [{"message": {"content": "Alex"}}], "usage": {"prompt_tokens": 10}},
        ),
        (
            "atomics.providers.together:TogetherProvider",
            {"api_key": "fake"},
            {"choices": [{"message": {"content": "Alex"}}], "usage": {"prompt_tokens": 10}},
        ),
    ],
)
async def test_openai_compat_providers_send_turns_as_chat_messages(
    cls_path: str, init_kwargs: dict, reply_json: dict
) -> None:
    import importlib

    mod_name, cls_name = cls_path.split(":")
    cls = getattr(importlib.import_module(mod_name), cls_name)

    reply = MagicMock()
    reply.raise_for_status = MagicMock()
    reply.json.return_value = reply_json
    client = AsyncMock()
    client.post = AsyncMock(return_value=reply)
    provider = cls(client=client, **init_kwargs)

    resp = await provider.generate_chat(HISTORY, system="Be brief.", max_tokens=64, temperature=0.0)

    body = client.post.call_args.kwargs["json"]
    assert body["messages"] == [{"role": "system", "content": "Be brief."}, *HISTORY]
    assert body["temperature"] == 0.0
    assert resp.text == "Alex"


@pytest.mark.asyncio
async def test_budget_guard_meters_chat_calls() -> None:
    rec = _Recorder()
    guard = EvalBudget(budget_limit_usd=1.0).new_guard()
    await GuardedProvider(rec, guard).generate_chat(HISTORY, system="s")
    assert rec.calls and guard.total_cost == 0.25


@pytest.mark.asyncio
async def test_trace_shows_history_apart_from_the_new_message() -> None:
    console = Console(record=True, width=200)
    trace.enable(console)
    try:
        await trace.traced(_Recorder()).generate_chat(HISTORY, system="Be brief.")
    finally:
        trace.disable()
    text = console.export_text()
    assert "history  2 earlier messages" in text
    assert "[assistant] Hi Alex." in text
    assert "new message" in text and "What is my name?" in text
