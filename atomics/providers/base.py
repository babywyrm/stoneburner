"""Abstract base for all LLM provider adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProviderResponse:
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    latency_ms: float
    estimated_cost_usd: float
    raw: dict | None = None


class BaseProvider(ABC):
    """Every provider adapter must implement these methods."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> ProviderResponse: ...

    @abstractmethod
    async def health_check(self) -> bool: ...
