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
