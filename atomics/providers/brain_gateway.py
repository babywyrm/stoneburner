"""Camazotz brain-gateway provider adapter.

Routes stoneburner benchmarks through camazotz's MCP brain gateway,
enabling comparative benchmarking of models controlled by camazotz's
provider-switching layer (cloud, local, bedrock, openai).
"""

from __future__ import annotations

import json
import time

import httpx

from atomics.providers.base import BaseProvider, ProviderResponse, compute_tps


class BrainGatewayProvider(BaseProvider):
    """Benchmark against a camazotz brain-gateway instance via MCP JSON-RPC."""

    def __init__(
        self,
        url: str = "http://localhost:8080",
        *,
        default_model: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._default_model = default_model
        self._client = client or httpx.AsyncClient()

    @property
    def name(self) -> str:
        return "brain-gateway"

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
        # temperature is accepted for interface parity but not enforceable here:
        # the brain-gateway controls sampling server-side via its own provider
        # config, so the ask_agent RPC has no per-call temperature field.
        effective_model = model or self._default_model

        if effective_model:
            await self._switch_model(effective_model)

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "config.ask_agent",
                "arguments": {"question": prompt},
            },
        }

        t0 = time.monotonic()
        try:
            resp = await self._client.post(
                f"{self._url}/mcp",
                json=payload,
                timeout=120.0,
            )
            resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot connect to brain-gateway at {self._url} — is it running?"
            ) from exc

        latency_ms = (time.monotonic() - t0) * 1000
        data = resp.json()

        if "error" in data:
            err = data["error"]
            raise RuntimeError(
                f"brain-gateway RPC error {err.get('code')}: {err.get('message')}"
            )

        content = data.get("result", {}).get("content", [])
        if not content:
            raise RuntimeError("brain-gateway returned empty content")

        inner = json.loads(content[0].get("text", "{}"))
        text = inner.get("answer", inner.get("ai_analysis", inner.get("summary", "")))

        usage = inner.get("_usage", {})
        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)
        cost = usage.get("cost_usd", 0.0)
        model_used = usage.get("model", effective_model or "unknown")

        tps = compute_tps(out, latency_ms / 1000)

        return ProviderResponse(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=inp + out,
            model=model_used,
            latency_ms=round(latency_ms, 2),
            estimated_cost_usd=cost,
            tokens_per_second=tps,
            raw=data,
        )

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(
                f"{self._url}/health",
                timeout=5.0,
            )
            return resp.status_code == 200
        except Exception:
            return False

    async def get_config(self) -> dict:
        """Fetch the current brain-gateway runtime config."""
        resp = await self._client.get(f"{self._url}/config", timeout=5.0)
        resp.raise_for_status()
        return resp.json()

    async def _switch_model(self, model: str) -> None:
        """Switch the brain-gateway's active model before generating."""
        try:
            await self._client.put(
                f"{self._url}/config",
                json={"model": model},
                timeout=10.0,
            )
        except httpx.HTTPError:
            pass
