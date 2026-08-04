"""Spend, rate, and failure ceilings for eval suites.

Benchmark runs have always been guarded: `LoopEngine` builds a `RateBudgetGuard`
from the tier profile, so a run stops when it hits the tier's dollar ceiling,
hourly token cap, or consecutive-failure threshold. Eval suites had no such
ceiling on any path. `POST /api/v1/evals` therefore let an authenticated caller
spend against provider accounts until the accounts themselves said no, and an
adversarial run with `--runs N` and extra judges is the most expensive thing
this tool does.

Rather than thread a guard parameter through the eighteen `provider.generate`
call sites spread across runners, judges, and scorers, the provider itself is
wrapped. Every suite is covered by construction, including suites not written
yet, and — importantly — judge traffic is covered too, since that is where
multi-judge consensus actually spends.

The model and judge should share one budget via `guarded_pair`. Two independent
ceilings would let a run cost twice what the operator asked for.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from atomics.core.guard import GuardConfig, RateBudgetGuard
from atomics.providers.base import BaseProvider, ProviderResponse

logger = logging.getLogger(__name__)


class EvalBudgetExceededError(RuntimeError):
    """An eval suite hit a ceiling and stopped rather than keep spending."""


@dataclass(frozen=True)
class EvalBudget:
    """Ceilings for one eval run.

    Deliberately more generous than a benchmark tier: an eval sweep is a
    deliberate act with a known shape, not a background loop. The point is to
    bound the damage from a runaway or hostile caller, not to second-guess an
    operator who knows what they are running.
    """

    budget_limit_usd: float = 10.0
    max_tokens_per_hour: int = 2_000_000
    max_requests_per_minute: int = 120
    circuit_breaker_threshold: int = 10

    def __post_init__(self) -> None:
        if self.budget_limit_usd <= 0:
            raise ValueError(
                f"budget_limit_usd must be positive, got {self.budget_limit_usd}"
            )
        if self.max_tokens_per_hour < 1:
            raise ValueError(
                f"max_tokens_per_hour must be positive, got {self.max_tokens_per_hour}"
            )
        if self.max_requests_per_minute < 1:
            raise ValueError(
                "max_requests_per_minute must be positive, got "
                f"{self.max_requests_per_minute}"
            )

    def to_guard_config(self) -> GuardConfig:
        return GuardConfig(
            max_tokens_per_hour=self.max_tokens_per_hour,
            max_requests_per_minute=self.max_requests_per_minute,
            budget_limit_usd=self.budget_limit_usd,
            circuit_breaker_threshold=self.circuit_breaker_threshold,
        )

    def new_guard(self) -> RateBudgetGuard:
        return RateBudgetGuard(self.to_guard_config())


class GuardedProvider(BaseProvider):
    """A provider that consults a `RateBudgetGuard` around every call.

    Requests-per-minute pressure is transient, so it waits. Everything else —
    the dollar ceiling, the hourly token cap, an open circuit breaker — is
    terminal and raises `EvalBudgetExceededError`, because the wait for those is
    either unbounded or an hour, and a suite that silently stalls for an hour
    is worse than one that stops and says why.
    """

    # Bounds a pathological wait even if the guard keeps reporting pressure.
    MAX_WAIT_SECONDS = 90.0
    # Wake a moment after the window closes rather than exactly on it. The
    # guard prunes entries strictly older than 60s, so sleeping precisely the
    # reported wait lands on the boundary, prunes nothing, and reports pressure
    # again. Real sleeps usually overshoot enough to hide this; depending on
    # scheduling jitter for correctness is not a plan.
    WAIT_MARGIN_SECONDS = 0.25

    def __init__(self, inner: BaseProvider, guard: RateBudgetGuard) -> None:
        self._inner = inner
        self._guard = guard

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def default_model(self) -> str | None:
        return self._inner.default_model

    @property
    def supports_tools(self) -> bool:  # type: ignore[override]
        return self._inner.supports_tools

    @property
    def guard(self) -> RateBudgetGuard:
        return self._guard

    @property
    def inner(self) -> BaseProvider:
        return self._inner

    async def health_check(self) -> bool:
        # Not metered: a health check is not model traffic and costs nothing.
        return await self._inner.health_check()

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
        await self._await_capacity()
        return await self._metered(
            self._inner.generate(
                prompt,
                system=system,
                model=model,
                max_tokens=max_tokens,
                thinking=thinking,
                thinking_budget=thinking_budget,
                temperature=temperature,
            )
        )

    async def generate_with_tools(
        self,
        prompt: str,
        *,
        tools: Sequence[dict],
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
        injected_tool_output: str | None = None,
    ) -> ProviderResponse:
        await self._await_capacity()
        return await self._metered(
            self._inner.generate_with_tools(
                prompt,
                tools=tools,
                system=system,
                model=model,
                max_tokens=max_tokens,
                injected_tool_output=injected_tool_output,
            )
        )

    async def _metered(self, call: object) -> ProviderResponse:
        """Await a provider call and record what it consumed, success or not.

        A failure is recorded with zero usage so the circuit breaker still sees
        it. Without that, a provider erroring on every request would spin
        forever instead of tripping.
        """
        try:
            response: ProviderResponse = await call  # type: ignore[misc]
        except Exception:
            self._guard.record_request(0, 0.0, success=False)
            raise
        self._guard.record_request(
            response.total_tokens,
            response.estimated_cost_usd,
            success=True,
        )
        return response

    async def _await_capacity(self) -> None:
        waited = 0.0
        while True:
            allowed, reason = self._guard.can_proceed()
            if allowed:
                return
            if not self._is_transient(reason):
                raise EvalBudgetExceededError(
                    f"eval stopped: {reason} "
                    f"(spent ${self._guard.total_cost:.4f})"
                )
            wait = max(self._guard.seconds_until_allowed(), 0.0)
            wait += self.WAIT_MARGIN_SECONDS
            if waited + wait > self.MAX_WAIT_SECONDS:
                raise EvalBudgetExceededError(
                    f"eval stopped: {reason}, and waiting for capacity would "
                    f"exceed {self.MAX_WAIT_SECONDS:.0f}s"
                )
            logger.info("Eval guard: %s — waiting %.1fs", reason, wait)
            await asyncio.sleep(wait)
            waited += wait

    @staticmethod
    def _is_transient(reason: str) -> bool:
        """Only per-minute request pressure clears on its own quickly."""
        return reason.startswith("rate limit")


def share_budget(
    budget: EvalBudget | None,
    *providers: BaseProvider,
) -> tuple[BaseProvider, ...]:
    """Wrap every provider in one run against a single shared budget.

    Variadic because a run is rarely two providers: `--extra-judges` adds one
    per consensus judge, and giving each its own ceiling would let a run cost
    several times what the operator asked for.

    Returns the providers untouched when `budget` is None, which is how the CLI
    keeps its existing unmetered behavior unless asked for a ceiling.
    """
    if budget is None:
        return providers
    guard = budget.new_guard()
    return tuple(GuardedProvider(provider, guard) for provider in providers)
