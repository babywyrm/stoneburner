"""AWS Bedrock provider adapter — scaffolded for Phase 2.

Implements the same BaseProvider interface as Claude.
Requires: pip install atomics[bedrock]
"""

from __future__ import annotations

from atomics.providers.base import BaseProvider, ProviderResponse


class BedrockProvider(BaseProvider):
    """Bedrock Claude adapter — Phase 2 implementation."""

    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    ) -> None:
        self._region = region
        self._model_id = model_id

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
        raise NotImplementedError(
            "Bedrock provider is scaffolded for Phase 2. "
            "Install atomics[bedrock] and implement invoke_model calls."
        )

    async def health_check(self) -> bool:
        raise NotImplementedError("Bedrock health check not yet implemented.")
