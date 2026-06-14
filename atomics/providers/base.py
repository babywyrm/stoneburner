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
    # How tokens_per_second was measured, so cross-provider comparisons are honest:
    #   "wall_clock"  — total output tokens / end-to-end request time (includes
    #                   network + queue + prompt processing; API providers).
    #   "generation"  — total output tokens / pure decode time (local providers
    #                   that report a generation duration, e.g. Ollama eval_duration).
    tps_basis: str = "wall_clock"
    thinking_tokens: int = 0
    thinking_text: str = ""
    # Prompt-caching usage (providers that support it; 0 elsewhere).
    # cache_read_tokens: input tokens served from cache (billed at a discount).
    # cache_write_tokens: input tokens written to cache (billed at a premium).
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    raw: dict | None = field(default=None, repr=False)


def compute_tps(output_tokens: int, seconds: float) -> float | None:
    """Tokens/second over the given elapsed seconds.

    Standardized across providers to use *total* output tokens (thinking tokens
    are real generated work) divided by the elapsed time. Returns None when the
    rate is undefined (no tokens or no measured time). The time *basis* differs
    per provider and is recorded separately in ProviderResponse.tps_basis.
    """
    if seconds > 0 and output_tokens > 0:
        return round(output_tokens / seconds, 2)
    return None


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
        temperature: float | None = None,
    ) -> ProviderResponse: ...

    @abstractmethod
    async def health_check(self) -> bool: ...
