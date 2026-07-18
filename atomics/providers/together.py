"""Together AI provider adapter — OpenAI-compatible cloud inference.

Auth: TOGETHER_API_KEY environment variable.
Endpoint: https://api.together.xyz/v1/chat/completions
"""

from __future__ import annotations

import time

import httpx

from atomics.providers.base import BaseProvider, ProviderResponse, compute_tps

MODEL_PRICING: dict[str, tuple[float, float]] = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (0.88, 0.88),
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": (0.18, 0.18),
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": (0.88, 0.88),
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": (3.50, 3.50),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": (0.60, 0.60),
    "mistralai/Mistral-7B-Instruct-v0.3": (0.20, 0.20),
    "Qwen/Qwen2.5-72B-Instruct-Turbo": (1.20, 1.20),
    "Qwen/Qwen2.5-7B-Instruct-Turbo": (0.30, 0.30),
    "deepseek-ai/DeepSeek-R1": (3.00, 7.00),
    "deepseek-ai/DeepSeek-V3": (0.50, 0.90),
    "google/gemma-2-27b-it": (0.80, 0.80),
    "google/gemma-2-9b-it": (0.30, 0.30),
}

DEFAULT_PRICING = (1.00, 1.00)


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    inp_price, out_price = MODEL_PRICING.get(model, DEFAULT_PRICING)
    return (input_tokens * inp_price + output_tokens * out_price) / 1_000_000


class TogetherProvider(BaseProvider):
    """Together AI cloud inference via OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        api_key: str,
        default_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        *,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._default_model = default_model
        self._timeout = timeout
        self._client = client or httpx.AsyncClient()
        self._base_url = "https://api.together.xyz/v1"

    @property
    def name(self) -> str:
        return "together"

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
