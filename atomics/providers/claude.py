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


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    inp_price, out_price = MODEL_PRICING.get(model, DEFAULT_PRICING)
    return (input_tokens * inp_price + output_tokens * out_price) / 1_000_000


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
    ) -> ProviderResponse:
        model = model or self._default_model
        messages = [{"role": "user", "content": prompt}]

        t0 = time.monotonic()
        response = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or "You are a helpful assistant.",
            messages=messages,
        )
        latency = (time.monotonic() - t0) * 1000

        text = response.content[0].text if response.content else ""
        inp = response.usage.input_tokens
        out = response.usage.output_tokens

        return ProviderResponse(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=inp + out,
            model=model,
            latency_ms=round(latency, 2),
            estimated_cost_usd=round(_estimate_cost(model, inp, out), 6),
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    async def health_check(self) -> bool:
        try:
            resp = await self.generate("Say OK.", max_tokens=8)
            return len(resp.text) > 0
        except Exception:
            return False
