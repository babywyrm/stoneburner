"""AWS Bedrock provider adapter using the Converse API.

Requires: pip install atomics[bedrock]  (adds boto3)
Auth: uses standard AWS credential chain (env vars, ~/.aws/credentials, IAM role).
"""

from __future__ import annotations

import time

from atomics.providers.base import BaseProvider, ProviderResponse

BEDROCK_PRICING: dict[str, tuple[float, float]] = {
    "us.anthropic.claude-sonnet-4-6": (3.0, 15.0),
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": (1.0, 5.0),
    "us.anthropic.claude-opus-4-6-v1": (5.0, 25.0),
    "us.anthropic.claude-sonnet-4-20250514-v1:0": (3.0, 15.0),
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": (3.0, 15.0),
    "anthropic.claude-sonnet-4-20250514-v1:0": (3.0, 15.0),
    "anthropic.claude-3-5-sonnet-20241022-v2:0": (3.0, 15.0),
    "anthropic.claude-3-5-haiku-20241022-v1:0": (0.80, 4.0),
}

DEFAULT_PRICING = (3.0, 15.0)


def _estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    inp_price, out_price = BEDROCK_PRICING.get(model_id, DEFAULT_PRICING)
    return (input_tokens * inp_price + output_tokens * out_price) / 1_000_000


class BedrockProvider(BaseProvider):
    """Bedrock Claude adapter using the Converse API."""

    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = "us.anthropic.claude-sonnet-4-6",
        *,
        client: object | None = None,
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

    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> ProviderResponse:
        import asyncio

        model_id = model or self._model_id
        loop = asyncio.get_running_loop()

        t0 = time.monotonic()
        response = await loop.run_in_executor(
            None,
            lambda: self._converse(prompt, system=system, model_id=model_id, max_tokens=max_tokens),
        )
        latency = (time.monotonic() - t0) * 1000

        text = ""
        for block in response.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                text += block["text"]

        usage = response.get("usage", {})
        inp = usage.get("inputTokens", 0)
        out = usage.get("outputTokens", 0)

        return ProviderResponse(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=inp + out,
            model=model_id,
            latency_ms=round(latency, 2),
            estimated_cost_usd=round(_estimate_cost(model_id, inp, out), 6),
            raw=response,
        )

    def _converse(self, prompt: str, *, system: str, model_id: str, max_tokens: int) -> dict:
        kwargs: dict = {
            "modelId": model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if system:
            kwargs["system"] = [{"text": system}]

        return self._client.converse(**kwargs)

    async def health_check(self) -> bool:
        try:
            resp = await self.generate("Say OK.", max_tokens=8)
            return len(resp.text) > 0
        except Exception:
            return False
