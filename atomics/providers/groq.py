"""Groq provider adapter — fast inference via OpenAI-compatible API.

Auth: GROQ_API_KEY environment variable.
Endpoint: https://api.groq.com/openai/v1/chat/completions
"""

from __future__ import annotations

import time

import httpx

from atomics.providers.base import BaseProvider, ProviderResponse, compute_tps

MODEL_PRICING: dict[str, tuple[float, float]] = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.2-1b-preview": (0.04, 0.04),
    "llama-3.2-3b-preview": (0.06, 0.06),
    "llama-3.2-11b-vision-preview": (0.18, 0.18),
    "llama-3.2-90b-vision-preview": (0.90, 0.90),
    "gemma2-9b-it": (0.20, 0.20),
    "mixtral-8x7b-32768": (0.24, 0.24),
    "qwen-qwq-32b": (0.29, 0.39),
    "deepseek-r1-distill-llama-70b": (0.75, 0.99),
    "meta-llama/llama-4-scout-17b-16e-instruct": (0.11, 0.34),
    "meta-llama/llama-4-maverick-17b-128e-instruct": (0.50, 0.77),
}

DEFAULT_PRICING = (0.50, 0.50)


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    inp_price, out_price = MODEL_PRICING.get(model, DEFAULT_PRICING)
    return (input_tokens * inp_price + output_tokens * out_price) / 1_000_000


class GroqProvider(BaseProvider):
    """Groq cloud inference via OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        api_key: str,
        default_model: str = "llama-3.3-70b-versatile",
        *,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._default_model = default_model
        self._timeout = timeout
        self._client = client or httpx.AsyncClient()
        self._base_url = "https://api.groq.com/openai/v1"

    @property
    def name(self) -> str:
        return "groq"

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
            raw=data,
        )

    async def health_check(self) -> bool:
        try:
            resp = await self.generate("Say OK.", max_tokens=8)
            return len(resp.text) > 0
        except Exception:
            return False
