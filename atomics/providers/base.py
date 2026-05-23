"""Abstract base for all LLM provider adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ProviderResponse:
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    latency_ms: float
    estimated_cost_usd: float
    tokens_per_second: float | None = None
    thinking_tokens: int = 0
    thinking_text: str = ""
    raw: dict | None = field(default=None, repr=False)


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
        thinking: bool | None = None,
        thinking_budget: int | None = None,
    ) -> ProviderResponse: ...

    @abstractmethod
    async def health_check(self) -> bool: ...
