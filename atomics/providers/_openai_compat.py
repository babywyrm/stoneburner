"""Shared tool-calling implementation for OpenAI-compatible HTTP providers.

vllm, llamacpp, groq, together and gemini all POST to a `/chat/completions`
endpoint with the same request and response shape, differing only in base URL,
auth headers, and cost table. Implementing `generate_with_tools` five times would
mean five places for the request body to drift; this mixin implements it once
against the attributes all five already define.

`HostedChatProvider` adds the one `generate()` that groq and together share
byte for byte. vllm, llamacpp and gemini keep their own: vLLM's reasoning-content
accounting and Gemini's request shape differ in ways that matter.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import httpx

from atomics.providers._tool_dialects import (
    openai_tool_payload,
    parse_openai_tool_calls,
)
from atomics.providers.base import BaseProvider, ProviderResponse, compute_tps
from atomics.providers.effort import apply_chat_effort, normalize_effort

# See _INJECTED_CALL_ID in providers/openai.py: a tool message's tool_call_id has
# to match a preceding assistant call, so both sides need the same constant.
_INJECTED_CALL_ID = "call_injected"


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

    def _augment_tool_body(
        self,
        body: dict[str, Any],
        *,
        model: str,
        thinking: bool | None,
        thinking_budget: int | None,
        effort: str | None,
    ) -> dict[str, Any] | None:
        """Optional extra chat-body keys. vLLM adds Qwen template fields."""
        del body, model, thinking, thinking_budget, effort
        return None

    async def generate_with_tools(
        self,
        prompt: str,
        *,
        tools: Sequence[dict],
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
        injected_tool_output: str | None = None,
        thinking: bool | None = None,
        thinking_budget: int | None = None,
        effort: str | None = None,
        reasoning_mode: str | None = None,
    ) -> ProviderResponse:
        del reasoning_mode
        this: Any = self
        model = model or this._default_model

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        if injected_tool_output is not None:
            # Indirect injection: the attack arrives as the result of a tool the
            # model appears to have already called.
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": _INJECTED_CALL_ID,
                            "type": "function",
                            "function": {
                                "name": "list_files",
                                "arguments": '{"directory": "."}',
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _INJECTED_CALL_ID,
                    "content": injected_tool_output,
                }
            )

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
            "tools": openai_tool_payload(list(tools)),
        }
        reasoning_request = apply_chat_effort(body, effort)
        extra = self._augment_tool_body(
            body,
            model=model,
            thinking=thinking,
            thinking_budget=thinking_budget,
            effort=effort,
        )
        if extra:
            reasoning_request = {**(reasoning_request or {}), **extra}

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
            effort=normalize_effort(effort),
            reasoning_request=reasoning_request,
        )


class HostedChatProvider(OpenAICompatibleTools, BaseProvider):
    """A bearer-token cloud API that speaks plain Chat Completions."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        default_model: str,
        timeout: float,
        client: httpx.AsyncClient | None,
    ) -> None:
        self._name = name
        self._base_url = base_url
        self._api_key = api_key
        self._default_model = default_model
        self._timeout = timeout
        self._client = client or httpx.AsyncClient()

    @property
    def name(self) -> str:
        return self._name

    @property
    def default_model(self) -> str:
        return self._default_model

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
        thinking: bool | None = None,
        thinking_budget: int | None = None,
        temperature: float | None = None,
        effort: str | None = None,
        reasoning_mode: str | None = None,
    ) -> ProviderResponse:
        model = model or self._default_model
        _ = reasoning_mode

        messages = [
            {"role": "system", "content": system or "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        body: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if temperature is not None:
            body["temperature"] = temperature
        reasoning_request = apply_chat_effort(body, effort)

        t0 = time.monotonic()
        response = await self._client.post(
            f"{self._base_url}/chat/completions",
            json=body,
            headers=self._headers(),
            timeout=self._timeout,
        )
        response.raise_for_status()
        latency_ms = round((time.monotonic() - t0) * 1000, 2)

        data = response.json()
        choice = data["choices"][0] if data.get("choices") else {}
        text = choice.get("message", {}).get("content", "") or ""
        usage = data.get("usage", {})
        inp = usage.get("prompt_tokens", 0)
        out = usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", inp + out)

        tps = compute_tps(out, latency_ms / 1000)

        return ProviderResponse(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=total,
            model=model,
            latency_ms=latency_ms,
            estimated_cost_usd=round(self._tool_cost(model, inp, out), 6),
            tokens_per_second=tps,
            tps_basis="wall_clock",
            raw=data,
            effort=normalize_effort(effort),
            reasoning_request=reasoning_request,
        )

    async def health_check(self) -> bool:
        try:
            resp = await self.generate("Say OK.", max_tokens=8)
            return len(resp.text) > 0
        except Exception:
            return False
