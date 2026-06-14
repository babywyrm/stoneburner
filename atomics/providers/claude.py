"""Claude (Anthropic API) provider adapter."""

from __future__ import annotations

import time

import anthropic

from atomics.providers import pricing
from atomics.providers.base import BaseProvider, ProviderResponse, compute_tps

# Pricing per 1M tokens (input / output). Sourced from the central pricing
# module; re-exported here for backward compatibility.
MODEL_PRICING = pricing.CLAUDE_PRICING
DEFAULT_PRICING = pricing.CLAUDE_DEFAULT


def _estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    return pricing.estimate_cost(
        model,
        input_tokens,
        output_tokens,
        table=MODEL_PRICING,
        default=DEFAULT_PRICING,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )


_DEFAULT_THINKING_BUDGET = 10_000


class ClaudeProvider(BaseProvider):
    def __init__(
        self,
        api_key: str,
        default_model: str = "claude-sonnet-4-20250514",
        *,
        client: object | None = None,
    ) -> None:
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key)
        self._default_model = default_model

    @property
    def name(self) -> str:
        return "claude"

    @property
    def default_model(self) -> str | None:
        return self._default_model

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
    ) -> ProviderResponse:
        model = model or self._default_model
        messages = [{"role": "user", "content": prompt}]

        use_thinking = thinking if thinking is not None else False

        kwargs: dict = {
            "model": model,
            "messages": messages,
        }

        if use_thinking:
            budget = thinking_budget or _DEFAULT_THINKING_BUDGET
            kwargs["max_tokens"] = max_tokens + budget
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            if system:
                kwargs["system"] = system
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["system"] = system or "You are a helpful assistant."
            # Anthropic requires temperature=1 when extended thinking is enabled,
            # so only forward an explicit temperature when thinking is off.
            if temperature is not None:
                kwargs["temperature"] = temperature

        t0 = time.monotonic()
        response = await self._client.messages.create(**kwargs)
        latency = (time.monotonic() - t0) * 1000

        text = ""
        thinking_text = ""
        for block in response.content:
            if getattr(block, "type", "") == "thinking":
                thinking_text += getattr(block, "thinking", "")
            elif getattr(block, "type", "") == "text":
                text += getattr(block, "text", "")

        if not text and response.content:
            text = getattr(response.content[0], "text", "")

        inp = response.usage.input_tokens
        out = response.usage.output_tokens
        # Anthropic bills extended-thinking tokens as ordinary output and does not
        # expose a separate thinking-token count, so this is 0 in production; the
        # getattr keeps the value truthful if a future/usage object ever reports one.
        thinking_tokens = getattr(response.usage, "thinking_tokens", 0) or 0
        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(response.usage, "cache_creation_input_tokens", 0) or 0

        tps = compute_tps(out, latency / 1000)

        return ProviderResponse(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=inp + out,
            model=model,
            latency_ms=round(latency, 2),
            estimated_cost_usd=round(
                _estimate_cost(model, inp, out, cache_read, cache_write), 6
            ),
            tokens_per_second=round(tps, 2) if tps is not None else None,
            thinking_tokens=thinking_tokens,
            thinking_text=thinking_text,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    async def health_check(self) -> bool:
        try:
            resp = await self.generate("Say OK.", max_tokens=8)
            return len(resp.text) > 0
        except Exception:
            return False
