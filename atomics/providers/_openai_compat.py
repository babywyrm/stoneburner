"""Shared tool-calling implementation for OpenAI-compatible HTTP providers.

vllm, llamacpp, groq, together and gemini all POST to a `/chat/completions`
endpoint with the same request and response shape, differing only in base URL,
auth headers, and cost table. Implementing `generate_with_tools` five times would
mean five places for the request body to drift; this mixin implements it once
against the attributes all five already define.

It deliberately does not touch `generate()`. Those implementations differ in ways
that matter — vLLM's reasoning-content accounting, per-provider cost tables — and
unifying them is a separate change with its own regression surface.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Protocol

import httpx

from atomics.providers._tool_dialects import (
    openai_tool_payload,
    parse_openai_tool_calls,
)
from atomics.providers.base import ProviderResponse, compute_tps

# See _INJECTED_CALL_ID in providers/openai.py: a tool message's tool_call_id has
# to match a preceding assistant call, so both sides need the same constant.
_INJECTED_CALL_ID = "call_injected"


class _CompatProvider(Protocol):
    """The attributes this mixin relies on, all defined by every provider using it."""

    _client: Any
    _base_url: str
    _default_model: str
    _timeout: float


class OpenAICompatibleTools:
    """Implements `generate_with_tools` for `/chat/completions` providers."""

    supports_tools = True

    # Appended to _base_url. llamacpp mounts the OpenAI surface under /v1 while
    # the others fold that into their base URL already.
    _tool_path = "/chat/completions"

    def _tool_headers(self) -> dict[str, str]:
        """Auth headers for the tool request.

        Reuses the provider's own `_headers()` where it has one; llamacpp is
        unauthenticated and has none.
        """
        headers = getattr(self, "_headers", None)
        if callable(headers):
            return dict(headers())
        return {"Content-Type": "application/json"}

    def _tool_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Cost for a tool request. Self-hosted providers override to stay at zero."""
        return 0.0

    async def generate_with_tools(
        self,
        prompt: str,
        *,
        tools: Sequence[dict],
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
        injected_tool_output: str | None = None,
    ) -> ProviderResponse:
        this: Any = self
        model = model or this._default_model

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        if injected_tool_output is not None:
            # Indirect injection: the attack arrives as the result of a tool the
            # model appears to have already called.
            messages.append({
                "role": "assistant",
                "tool_calls": [{
                    "id": _INJECTED_CALL_ID,
                    "type": "function",
                    "function": {
                        "name": "list_files",
                        "arguments": '{"directory": "."}',
                    },
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": _INJECTED_CALL_ID,
                "content": injected_tool_output,
            })

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
            "tools": openai_tool_payload(list(tools)),
        }

        url = f"{this._base_url}{self._tool_path}"
        t0 = time.monotonic()
        try:
            response = await this._client.post(
                url,
                json=body,
                headers=self._tool_headers(),
                timeout=this._timeout,
            )
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot connect to endpoint at {this._base_url} — is it running?"
            ) from exc
        latency_ms = round((time.monotonic() - t0) * 1000, 2)

        data = response.json()
        choice = data["choices"][0] if data.get("choices") else {}
        message = choice.get("message") or {}
        text = message.get("content") or ""
        usage = data.get("usage", {})
        inp = usage.get("prompt_tokens", 0)
        out = usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", inp + out)

        return ProviderResponse(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=total,
            model=model,
            latency_ms=latency_ms,
            estimated_cost_usd=round(self._tool_cost(model, inp, out), 6),
            tokens_per_second=compute_tps(out, latency_ms / 1000),
            tps_basis="wall_clock",
            raw=data,
            finish_reason=choice.get("finish_reason"),
            tool_calls=parse_openai_tool_calls(message),
        )
