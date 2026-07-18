"""llama.cpp server provider adapter — direct local inference without Ollama.

Targets the llama.cpp HTTP server (llama-server / llama-cpp-python),
which exposes an OpenAI-compatible /v1/chat/completions endpoint.

Start the server: llama-server -m model.gguf --port 8080
Auth: none (local server).
"""

from __future__ import annotations

import time

import httpx

from atomics.providers.base import BaseProvider, ProviderResponse, compute_tps


class LlamaCppProvider(BaseProvider):
    """llama.cpp server adapter via OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        default_model: str = "local",
        *,
        timeout: float = 300.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._timeout = timeout
        self._client = client or httpx.AsyncClient()

    @property
    def name(self) -> str:
        return "llamacpp"

    @property
    def default_model(self) -> str:
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

        messages = [
            {"role": "system", "content": system or "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        body: dict = {
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if temperature is not None:
            body["temperature"] = temperature

        t0 = time.monotonic()
        try:
            response = await self._client.post(
                f"{self._base_url}/v1/chat/completions",
                json=body,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot connect to llama.cpp server at {self._base_url} — "
                f"start with: llama-server -m model.gguf --port 8080"
            ) from exc

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
            estimated_cost_usd=0.0,
            tokens_per_second=tps,
            tps_basis="wall_clock",
            raw=data,
        )

    async def health_check(self) -> bool:
        try:
            response = await self._client.get(
                f"{self._base_url}/health",
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception:
            return False
