"""Google Gemini provider adapter via the OpenAI-compatible API.

Auth: GEMINI_API_KEY environment variable.
Endpoint: https://generativelanguage.googleapis.com/v1beta/openai/chat/completions

Google's Gemini API supports the OpenAI Chat Completions format, so we use
httpx directly rather than adding a google-specific SDK dependency.
"""

from __future__ import annotations

import time

import httpx

from atomics.providers.base import BaseProvider, ProviderResponse, compute_tps

MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-flash-lite-preview-06-17": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.0),
    "gemini-1.5-flash": (0.075, 0.30),
}

DEFAULT_PRICING = (0.15, 0.60)


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    inp_price, out_price = MODEL_PRICING.get(model, DEFAULT_PRICING)
    return (input_tokens * inp_price + output_tokens * out_price) / 1_000_000


class GeminiProvider(BaseProvider):
    """Google Gemini via OpenAI-compatible Chat Completions endpoint."""

    def __init__(
        self,
        api_key: str,
        default_model: str = "gemini-2.5-flash",
        *,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._default_model = default_model
        self._timeout = timeout
        self._client = client or httpx.AsyncClient()
        self._base_url = "https://generativelanguage.googleapis.com/v1beta/openai"

    @property
    def name(self) -> str:
        return "gemini"

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
    ) -> ProviderResponse:
        model = model or self._default_model

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

        thinking_tokens = 0
        details = usage.get("completion_tokens_details", {})
        if details:
            thinking_tokens = details.get("reasoning_tokens", 0)

        return ProviderResponse(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=total,
            model=model,
            latency_ms=latency_ms,
            estimated_cost_usd=round(_estimate_cost(model, inp, out), 6),
            tokens_per_second=tps,
            tps_basis="wall_clock",
            thinking_tokens=thinking_tokens,
            raw=data,
        )

    async def health_check(self) -> bool:
        try:
            resp = await self.generate("Say OK.", max_tokens=8)
            return len(resp.text) > 0
        except Exception:
            return False
