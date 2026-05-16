"""Ollama (local inference) provider adapter."""

from __future__ import annotations

import httpx

from atomics.providers.base import BaseProvider, ProviderResponse


class OllamaProvider(BaseProvider):
    def __init__(
        self,
        host: str = "http://localhost:11434",
        default_model: str = "qwen2.5:7b",
        *,
        client: object | None = None,
    ) -> None:
        self._host = host.rstrip("/")
        self._default_model = default_model
        self._client = client or httpx.AsyncClient()

    @property
    def name(self) -> str:
        return "ollama"

    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> ProviderResponse:
        model = model or self._default_model

        try:
            response = await self._client.post(
                f"{self._host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "system": system or "You are a helpful assistant.",
                },
                timeout=120.0,
            )
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self._host} — is it running?"
            ) from exc

        data = response.json()
        text = data.get("response", "")
        out = data.get("eval_count", 0)
        inp = data.get("prompt_eval_count", 0)

        eval_duration = data.get("eval_duration", 0)
        tps = out / (eval_duration / 1e9) if eval_duration > 0 and out > 0 else None

        total_duration = data.get("total_duration", 0)
        latency = total_duration / 1e6 if total_duration else 0.0

        return ProviderResponse(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=inp + out,
            model=model,
            latency_ms=round(latency, 2),
            estimated_cost_usd=0.0,
            tokens_per_second=round(tps, 2) if tps is not None else None,
            raw=data,
        )

    async def health_check(self) -> bool:
        try:
            response = await self._client.get(
                f"{self._host}/api/tags",
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception:
            return False
