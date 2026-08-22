"""Abstract base for all LLM provider adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from atomics.providers.outcomes import ProviderOutcome
from atomics.providers.toolcalls import ToolCall


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
    outcome: ProviderOutcome | None = None
    finish_reason: str | None = None
    # Structured calls the model asked to make. Empty for every text-only
    # response, which is all of them outside the toolcall suite.
    tool_calls: tuple[ToolCall, ...] = ()
    # Shared operator dials and the native payload actually sent. Empty
    # when the call used provider defaults.
    effort: str | None = None
    reasoning_mode: str | None = None
    reasoning_request: dict | None = field(default=None, repr=False)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "finish_reason" and "finish_reason" in self.__dict__:
            outcome = self.__dict__.get("outcome")
            if isinstance(outcome, ProviderOutcome) and value != outcome.finish_reason:
                raise ValueError("finish_reason conflicts with outcome.finish_reason")
        elif name == "outcome" and "outcome" in self.__dict__:
            finish_reason = self.__dict__.get("finish_reason")
            if isinstance(value, ProviderOutcome) and value.finish_reason != finish_reason:
                raise ValueError("finish_reason conflicts with outcome.finish_reason")
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        if self.outcome is None or self.outcome.finish_reason is None:
            return
        if self.finish_reason is None:
            self.finish_reason = self.outcome.finish_reason
        elif self.finish_reason != self.outcome.finish_reason:
            raise ValueError("finish_reason conflicts with outcome.finish_reason")


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

    # Whether this provider implements generate_with_tools. A plain class
    # attribute rather than an abstract property so the providers that do not
    # support tools need no change at all.
    supports_tools: bool = False

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def default_model(self) -> str | None:
        """The model this adapter uses when a call omits an explicit model.

        Used for self-judge detection and logging. Subclasses override to expose
        their configured default; the base returns None when it cannot be known.
        """
        return None

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
        effort: str | None = None,
        reasoning_mode: str | None = None,
    ) -> ProviderResponse: ...

    @abstractmethod
    async def health_check(self) -> bool: ...

    async def generate_with_tools(
        self,
        prompt: str,
        *,
        tools: Sequence[dict],
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
        injected_tool_output: str | None = None,
        thinking: bool | None = None,
        thinking_budget: int | None = None,
        effort: str | None = None,
        reasoning_mode: str | None = None,
    ) -> ProviderResponse:
        """Generate with tool schemas attached, returning any calls emitted.

        Deliberately concrete and raising, not abstract: an abstract method here
        would break every existing provider at instantiation. Callers must check
        `supports_tools` first — a silent empty result would be scored as
        resistance, which is exactly the confound the toolcall suite exists to
        avoid.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support tool calling. "
            "Check provider.supports_tools before calling this."
        )
