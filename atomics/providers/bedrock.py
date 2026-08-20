"""AWS Bedrock provider adapter using the Converse API.

Requires: pip install stoneburner-atomics[bedrock]  (adds boto3)
Auth: uses standard AWS credential chain (env vars, ~/.aws/credentials, IAM role).
"""

from __future__ import annotations

import time
from typing import Any, Protocol

from atomics.providers import pricing
from atomics.providers.base import BaseProvider, ProviderResponse, compute_tps
from atomics.providers.effort import claude_request, normalize_effort


class _BedrockRuntimeClient(Protocol):
    """Structural type for the boto3 bedrock-runtime client (no stubs ship)."""

    def converse(self, **kwargs: Any) -> dict: ...


# Pricing per 1M tokens (input / output). Sourced from the central pricing
# module; re-exported here for backward compatibility.
BEDROCK_PRICING = pricing.BEDROCK_PRICING
DEFAULT_PRICING = pricing.BEDROCK_DEFAULT


def _estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    return pricing.estimate_cost(
        model_id, input_tokens, output_tokens, table=BEDROCK_PRICING, default=DEFAULT_PRICING
    )


class BedrockProvider(BaseProvider):
    """Bedrock Claude adapter using the Converse API."""

    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = "us.anthropic.claude-sonnet-4-6",
        *,
        client: _BedrockRuntimeClient | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            try:
                import boto3
            except ImportError as exc:
                raise ImportError(
                    "boto3 is required for the Bedrock provider. "
                    "Install with: uv sync --extra bedrock"
                ) from exc
            self._client = boto3.client("bedrock-runtime", region_name=region)
        self._model_id = model_id
        self._region = region

    @property
    def name(self) -> str:
        return "bedrock"

    @property
    def default_model(self) -> str | None:
        return self._model_id

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
        import asyncio

        model_id = model or self._model_id
        _ = reasoning_mode
        loop = asyncio.get_running_loop()

        t0 = time.monotonic()
        response = await loop.run_in_executor(
            None,
            lambda: self._converse(
                prompt,
                system=system,
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking=thinking,
                thinking_budget=thinking_budget,
                effort=effort,
            ),
        )
        latency = (time.monotonic() - t0) * 1000

        text = ""
        for block in response.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                text += block["text"]

        usage = response.get("usage", {})
        inp = usage.get("inputTokens", 0)
        out = usage.get("outputTokens", 0)

        thinking_block, extra = claude_request(
            model=model_id,
            thinking=thinking,
            thinking_budget=thinking_budget,
            effort=effort,
        )
        reasoning_request = {**({"thinking": thinking_block} if thinking_block else {}), **extra}
        return ProviderResponse(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=inp + out,
            model=model_id,
            latency_ms=round(latency, 2),
            estimated_cost_usd=round(_estimate_cost(model_id, inp, out), 6),
            tokens_per_second=compute_tps(out, latency / 1000),
            raw=response,
            effort=normalize_effort(effort),
            reasoning_request=reasoning_request or None,
        )

    def _converse(
        self,
        prompt: str,
        *,
        system: str,
        model_id: str,
        max_tokens: int,
        temperature: float | None = None,
        thinking: bool | None = None,
        thinking_budget: int | None = None,
        effort: str | None = None,
    ) -> dict:
        inference_config: dict = {"maxTokens": max_tokens}
        if temperature is not None:
            inference_config["temperature"] = temperature
        kwargs: dict = {
            "modelId": model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": inference_config,
        }
        if system:
            kwargs["system"] = [{"text": system}]
        thinking_block, extra = claude_request(
            model=model_id,
            thinking=thinking,
            thinking_budget=thinking_budget,
            effort=effort,
        )
        fields: dict[str, object] = {}
        if thinking_block is not None:
            fields["thinking"] = thinking_block
        fields.update(extra)
        if fields:
            kwargs["additionalModelRequestFields"] = fields

        return self._client.converse(**kwargs)

    async def health_check(self) -> bool:
        try:
            resp = await self.generate("Say OK.", max_tokens=8)
            return len(resp.text) > 0
        except Exception:
            return False
