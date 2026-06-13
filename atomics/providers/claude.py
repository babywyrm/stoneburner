"""Claude (Anthropic API) provider adapter."""

from __future__ import annotations

import time

import anthropic

from atomics.providers.base import BaseProvider, ProviderResponse

# Pricing per 1M tokens (input / output)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-opus-4-20250514": (15.0, 75.0),
}

DEFAULT_PRICING = (3.0, 15.0)


# Anthropic prompt-caching multipliers (relative to the base input rate):
# cache writes bill at 1.25x, cache reads at 0.10x. Cached tokens are reported
# separately from input_tokens, so they are additive in the cost calculation.
_CACHE_WRITE_MULTIPLIER = 1.25
_CACHE_READ_MULTIPLIER = 0.10


def _estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    inp_price, out_price = MODEL_PRICING.get(model, DEFAULT_PRICING)
    cost = (
        input_tokens * inp_price
        + cache_write_tokens * inp_price * _CACHE_WRITE_MULTIPLIER
        + cache_read_tokens * inp_price * _CACHE_READ_MULTIPLIER
        + output_tokens * out_price
    )
    return cost / 1_000_000


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

    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
        thinking: bool | None = None,
        thinking_budget: int | None = None,
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
        thinking_tokens = getattr(response.usage, "thinking_tokens", 0) or 0
        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(response.usage, "cache_creation_input_tokens", 0) or 0

        visible_out = out - thinking_tokens
        tps = visible_out / (latency / 1000) if latency > 0 and visible_out > 0 else None

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
