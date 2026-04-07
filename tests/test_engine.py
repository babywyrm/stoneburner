"""Tests for the loop engine using a mock provider."""

import asyncio

import pytest

from atomics.core.guard import RateBudgetGuard
from atomics.models import BurnTier
from atomics.providers.base import BaseProvider, ProviderResponse


class FailingThenMockProvider(BaseProvider):
    """First generate() raises; subsequent calls delegate to a fresh mock response."""

    def __init__(self) -> None:
        self.attempts = 0

    @property
    def name(self) -> str:
        return "flaky"

    async def generate(self, prompt, *, system="", model=None, max_tokens=1024):
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("simulated provider failure")
        return ProviderResponse(
            text="ok",
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            model=model or "m",
            latency_ms=1.0,
            estimated_cost_usd=0.0,
        )

    async def health_check(self):
        return True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_runs_n_iterations(make_engine):
    engine, provider, repo = make_engine()
    await engine.run(max_iterations=5)
    assert provider.call_count == 5
    runs = repo.get_recent_runs(limit=1)
    assert runs[0]["total_tasks"] == 5
    assert runs[0]["successful_tasks"] == 5
    repo.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_respects_tier(make_engine):
    engine, _, repo = make_engine(tier=BurnTier.MEGA)
    assert engine.tier == BurnTier.MEGA
    await engine.run(max_iterations=1)
    repo.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_records_metrics(make_engine):
    engine, _, repo = make_engine()
    await engine.run(max_iterations=3)
    runs = repo.get_recent_runs(limit=1)
    assert runs[0]["total_tokens"] == 270  # 90 * 3
    assert runs[0]["total_cost_usd"] == pytest.approx(0.0015, abs=0.0001)
    repo.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_stops_on_budget(make_engine):
    engine, provider, repo = make_engine(budget=0.0008)
    await engine.run(max_iterations=100)
    assert provider.call_count < 100
    repo.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_logs_task_failure(make_engine, caplog):
    caplog.set_level("WARNING")
    engine, _, repo = make_engine()
    engine._provider = FailingThenMockProvider()  # noqa: SLF001 — intentional for failure-path coverage

    await engine.run(max_iterations=2)
    assert "FAIL" in caplog.text
    repo.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_guard_rate_limit_yields_then_continues(make_engine, monkeypatch):
    engine, provider, repo = make_engine()
    real_wait = asyncio.wait_for
    wait_calls: list[float | None] = []

    async def tracking_wait_for(coro, timeout=None):
        wait_calls.append(timeout)
        if len(wait_calls) == 1:
            coro.close()
            raise TimeoutError
        return await real_wait(coro, timeout=timeout)

    monkeypatch.setattr("atomics.core.engine.asyncio.wait_for", tracking_wait_for)

    cp_calls = {"n": 0}
    real_can = RateBudgetGuard.can_proceed

    def fake_can_proceed(self):
        cp_calls["n"] += 1
        if cp_calls["n"] == 1:
            return False, "rate limit (30 req/min)"
        return real_can(self)

    monkeypatch.setattr(RateBudgetGuard, "can_proceed", fake_can_proceed)

    await engine.run(max_iterations=1)
    assert provider.call_count == 1
    assert len(wait_calls) >= 1
    repo.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_stops_during_interval_wait(make_engine):
    engine, provider, repo = make_engine(interval=1)

    async def run_and_stop():
        await asyncio.sleep(0.05)
        engine.stop()

    await asyncio.gather(engine.run(max_iterations=50), run_and_stop())
    assert provider.call_count >= 1
    assert provider.call_count < 50
    repo.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_engine_stop_sets_shutdown_event(make_engine):
    engine, _, repo = make_engine()
    engine.stop()
    assert engine._shutdown.is_set()  # noqa: SLF001
    repo.close()

