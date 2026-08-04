"""Tests for eval spend ceilings.

Eval suites were the one remotely reachable path that spent with no ceiling
behind it, so these assert the boundary itself: that a run stops, that it stops
at the right point, and that the model and its judges are counted together.
"""

from __future__ import annotations

import asyncio

import pytest

from atomics.eval.budget import (
    BudgetMeter,
    EvalBudget,
    EvalBudgetExceededError,
    GuardedProvider,
    share_budget,
)
from atomics.providers.base import BaseProvider, ProviderResponse


class StubProvider(BaseProvider):
    """Records calls and returns a fixed cost per call."""

    supports_tools = True

    def __init__(
        self,
        *,
        cost: float = 1.0,
        tokens: int = 100,
        fail: bool = False,
        name: str = "stub",
    ) -> None:
        self._cost = cost
        self._tokens = tokens
        self._fail = fail
        self._name = name
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def default_model(self) -> str | None:
        return "stub-model"

    async def health_check(self) -> bool:
        return True

    def _response(self) -> ProviderResponse:
        return ProviderResponse(
            text="ok",
            input_tokens=self._tokens // 2,
            output_tokens=self._tokens // 2,
            total_tokens=self._tokens,
            model="stub-model",
            latency_ms=1.0,
            estimated_cost_usd=self._cost,
        )

    async def generate(self, prompt, **kwargs) -> ProviderResponse:
        self.calls += 1
        if self._fail:
            raise RuntimeError("provider exploded")
        return self._response()

    async def generate_with_tools(self, prompt, **kwargs) -> ProviderResponse:
        self.calls += 1
        if self._fail:
            raise RuntimeError("provider exploded")
        return self._response()


async def _call(provider: BaseProvider) -> ProviderResponse:
    return await provider.generate("hi")


class TestBudgetValidation:
    @pytest.mark.parametrize("value", [0, -1, -0.01])
    def test_a_non_positive_budget_is_rejected(self, value):
        with pytest.raises(ValueError, match="budget_limit_usd must be positive"):
            EvalBudget(budget_limit_usd=value)

    def test_non_positive_rate_limits_are_rejected(self):
        with pytest.raises(ValueError, match="max_tokens_per_hour"):
            EvalBudget(max_tokens_per_hour=0)
        with pytest.raises(ValueError, match="max_requests_per_minute"):
            EvalBudget(max_requests_per_minute=0)

    def test_the_guard_config_carries_every_ceiling(self):
        config = EvalBudget(
            budget_limit_usd=3.0,
            max_tokens_per_hour=111,
            max_requests_per_minute=7,
            circuit_breaker_threshold=2,
        ).to_guard_config()
        assert config.budget_limit_usd == 3.0
        assert config.max_tokens_per_hour == 111
        assert config.max_requests_per_minute == 7
        assert config.circuit_breaker_threshold == 2


class TestSpendCeiling:
    @pytest.mark.asyncio
    async def test_a_run_stops_once_the_budget_is_spent(self):
        inner = StubProvider(cost=1.0)
        guarded = GuardedProvider(inner, EvalBudget(budget_limit_usd=3.0).new_guard())

        for _ in range(3):
            await _call(guarded)

        with pytest.raises(EvalBudgetExceededError, match="budget exhausted"):
            await _call(guarded)

    @pytest.mark.asyncio
    async def test_the_provider_is_not_called_after_the_ceiling(self):
        inner = StubProvider(cost=5.0)
        guarded = GuardedProvider(inner, EvalBudget(budget_limit_usd=5.0).new_guard())

        await _call(guarded)
        with pytest.raises(EvalBudgetExceededError):
            await _call(guarded)

        assert inner.calls == 1, "the blocked call must not reach the provider"

    @pytest.mark.asyncio
    async def test_the_error_reports_what_was_spent(self):
        inner = StubProvider(cost=2.5)
        guarded = GuardedProvider(inner, EvalBudget(budget_limit_usd=2.0).new_guard())
        await _call(guarded)

        with pytest.raises(EvalBudgetExceededError, match=r"\$2\.5000"):
            await _call(guarded)

    @pytest.mark.asyncio
    async def test_a_run_under_budget_is_untouched(self):
        inner = StubProvider(cost=0.01)
        guarded = GuardedProvider(inner, EvalBudget(budget_limit_usd=10.0).new_guard())

        for _ in range(20):
            response = await _call(guarded)
            assert response.text == "ok"
        assert inner.calls == 20

    @pytest.mark.asyncio
    async def test_free_local_models_never_trip_the_ceiling(self):
        """Ollama and friends report zero cost; they must not be throttled."""
        inner = StubProvider(cost=0.0, tokens=10)
        guarded = GuardedProvider(inner, EvalBudget(budget_limit_usd=0.01).new_guard())

        for _ in range(50):
            await _call(guarded)
        assert inner.calls == 50


class TestTokenAndFailureCeilings:
    @pytest.mark.asyncio
    async def test_the_hourly_token_cap_stops_the_run(self):
        inner = StubProvider(cost=0.0, tokens=1000)
        guarded = GuardedProvider(
            inner,
            EvalBudget(budget_limit_usd=100.0, max_tokens_per_hour=2000).new_guard(),
        )
        await _call(guarded)
        await _call(guarded)

        with pytest.raises(EvalBudgetExceededError, match="hourly token cap"):
            await _call(guarded)

    @pytest.mark.asyncio
    async def test_repeated_provider_failures_trip_the_circuit(self):
        inner = StubProvider(fail=True)
        guarded = GuardedProvider(
            inner,
            EvalBudget(budget_limit_usd=100.0, circuit_breaker_threshold=3).new_guard(),
        )
        for _ in range(3):
            with pytest.raises(RuntimeError, match="provider exploded"):
                await _call(guarded)

        with pytest.raises(EvalBudgetExceededError, match="circuit breaker"):
            await _call(guarded)

    @pytest.mark.asyncio
    async def test_a_failure_still_counts_toward_the_circuit(self):
        """Without recording failures the breaker would never trip."""
        inner = StubProvider(fail=True)
        guarded = GuardedProvider(
            inner, EvalBudget(circuit_breaker_threshold=1).new_guard()
        )
        with pytest.raises(RuntimeError):
            await _call(guarded)
        assert guarded.guard.circuit_open


class TestSharedBudget:
    @pytest.mark.asyncio
    async def test_the_model_and_judge_draw_from_one_ceiling(self):
        model = StubProvider(cost=1.0, name="model")
        judge = StubProvider(cost=1.0, name="judge")
        g_model, g_judge = share_budget(EvalBudget(budget_limit_usd=2.0), model, judge)

        await _call(g_model)
        await _call(g_judge)

        # Two providers, one ceiling: the third call is refused regardless of
        # which one makes it.
        with pytest.raises(EvalBudgetExceededError):
            await _call(g_model)
        with pytest.raises(EvalBudgetExceededError):
            await _call(g_judge)

    @pytest.mark.asyncio
    async def test_extra_judges_share_the_same_ceiling(self):
        providers = [StubProvider(cost=1.0, name=f"p{i}") for i in range(4)]
        guarded = share_budget(EvalBudget(budget_limit_usd=2.0), *providers)

        await _call(guarded[0])
        await _call(guarded[1])
        with pytest.raises(EvalBudgetExceededError):
            await _call(guarded[3])

    def test_no_budget_returns_the_providers_untouched(self):
        model = StubProvider()
        judge = StubProvider()
        result = share_budget(None, model, judge)
        assert result == (model, judge)
        assert not isinstance(result[0], GuardedProvider)

    def test_share_budget_wraps_every_provider_given(self):
        guarded = share_budget(EvalBudget(), StubProvider(), StubProvider())
        assert all(isinstance(p, GuardedProvider) for p in guarded)


class TestTransparency:
    """The wrapper must be indistinguishable from the provider it wraps."""

    def test_the_name_passes_through(self):
        guarded = GuardedProvider(StubProvider(name="ollama"), EvalBudget().new_guard())
        assert guarded.name == "ollama"

    def test_the_default_model_passes_through(self):
        guarded = GuardedProvider(StubProvider(), EvalBudget().new_guard())
        assert guarded.default_model == "stub-model"

    def test_tool_support_passes_through(self):
        """Suites branch on this; reporting the wrapper's own value would break them."""
        guarded = GuardedProvider(StubProvider(), EvalBudget().new_guard())
        assert guarded.supports_tools is True

    def test_tool_support_is_false_when_the_inner_provider_lacks_it(self):
        inner = StubProvider()
        inner.supports_tools = False
        assert GuardedProvider(inner, EvalBudget().new_guard()).supports_tools is False

    @pytest.mark.asyncio
    async def test_health_checks_are_not_metered(self):
        inner = StubProvider(cost=1.0)
        guarded = GuardedProvider(inner, EvalBudget(budget_limit_usd=0.5).new_guard())
        assert await guarded.health_check() is True
        assert guarded.guard.total_cost == 0.0

    @pytest.mark.asyncio
    async def test_tool_calls_are_metered_too(self):
        inner = StubProvider(cost=1.0)
        guarded = GuardedProvider(inner, EvalBudget(budget_limit_usd=1.0).new_guard())
        await guarded.generate_with_tools("hi", tools=[])

        with pytest.raises(EvalBudgetExceededError):
            await guarded.generate_with_tools("hi", tools=[])

    @pytest.mark.asyncio
    async def test_the_response_is_returned_unchanged(self):
        guarded = GuardedProvider(StubProvider(tokens=42), EvalBudget().new_guard())
        response = await _call(guarded)
        assert response.total_tokens == 42
        assert response.model == "stub-model"

    def test_the_wrapped_provider_stays_reachable(self):
        inner = StubProvider()
        assert GuardedProvider(inner, EvalBudget().new_guard()).inner is inner

    def test_provider_specific_attributes_are_forwarded(self):
        """Callers reach past BaseProvider in practice.

        `sweep` reads `provider._host` to point a judge at the same Ollama
        instance. A wrapper that only forwards the declared interface would
        raise AttributeError there, which unit tests using a tidy stub would
        never surface.
        """
        inner = StubProvider()
        inner._host = "http://192.168.1.156:11434"
        guarded = GuardedProvider(inner, EvalBudget().new_guard())

        assert guarded._host == "http://192.168.1.156:11434"

    def test_an_genuinely_missing_attribute_still_raises(self):
        guarded = GuardedProvider(StubProvider(), EvalBudget().new_guard())
        with pytest.raises(AttributeError):
            guarded.no_such_attribute

    def test_forwarding_does_not_recurse_when_inner_is_absent(self):
        """__getattr__ reaching for _inner before it is set would loop forever."""
        orphan = GuardedProvider.__new__(GuardedProvider)
        with pytest.raises(AttributeError):
            orphan._inner


class TestBudgetMeter:
    """Sweeps build providers as they go, so the ceiling outlives each one."""

    def test_providers_built_at_different_times_share_one_ceiling(self):
        meter = BudgetMeter(EvalBudget(budget_limit_usd=2.0))
        first = meter.wrap(StubProvider(cost=1.0))
        second = meter.wrap(StubProvider(cost=1.0))

        assert first.guard is second.guard

    @pytest.mark.asyncio
    async def test_an_n_model_sweep_does_not_cost_n_ceilings(self):
        meter = BudgetMeter(EvalBudget(budget_limit_usd=2.0))
        models = [meter.wrap(StubProvider(cost=1.0)) for _ in range(5)]

        await _call(models[0])
        await _call(models[1])

        # The ceiling is for the sweep, not for each model in it.
        for model in models[2:]:
            with pytest.raises(EvalBudgetExceededError):
                await _call(model)

    def test_no_budget_means_providers_pass_through(self):
        meter = BudgetMeter(None)
        inner = StubProvider()

        assert meter.wrap(inner) is inner
        assert meter.active is False
        assert meter.guard is None

    def test_an_active_meter_reports_itself(self):
        meter = BudgetMeter(EvalBudget())
        assert meter.active is True
        assert meter.guard is not None


class FakeClock:
    """A monotonic clock the test advances by hand.

    The guard decides on elapsed time, so a fake `asyncio.sleep` that does not
    move the clock models a wait that accomplishes nothing — the guard would
    still report pressure on the next check and the run would abort. Advancing
    this alongside the sleep is what makes the test represent a real wait.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now


class TestRateLimitWaiting:
    @pytest.mark.asyncio
    async def test_per_minute_pressure_waits_rather_than_failing(self, monkeypatch):
        """Request pressure clears on its own, so it should not abort a run."""
        slept: list[float] = []
        clock = FakeClock()
        monkeypatch.setattr("atomics.core.guard.time", clock)

        async def fake_sleep(seconds: float) -> None:
            slept.append(seconds)
            clock.now += seconds

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        inner = StubProvider(cost=0.0)
        guarded = GuardedProvider(
            inner,
            EvalBudget(budget_limit_usd=100.0, max_requests_per_minute=2).new_guard(),
        )
        await _call(guarded)
        await _call(guarded)
        await _call(guarded)

        assert slept, "the third call should have waited for capacity"
        assert inner.calls == 3

    @pytest.mark.asyncio
    async def test_an_unbounded_wait_stops_the_run_instead(self, monkeypatch):
        async def fake_sleep(seconds: float) -> None:
            return None

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        inner = StubProvider(cost=0.0)
        guarded = GuardedProvider(
            inner,
            EvalBudget(budget_limit_usd=100.0, max_requests_per_minute=1).new_guard(),
        )
        guarded.MAX_WAIT_SECONDS = 0.0
        await _call(guarded)

        with pytest.raises(EvalBudgetExceededError, match="waiting for capacity"):
            await _call(guarded)

    def test_only_rate_limits_are_treated_as_transient(self):
        assert GuardedProvider._is_transient("rate limit (30 req/min)") is True
        assert GuardedProvider._is_transient("budget exhausted ($1 >= $1)") is False
        assert GuardedProvider._is_transient("hourly token cap (10/10)") is False
        assert GuardedProvider._is_transient("circuit breaker open (10 errors)") is False
