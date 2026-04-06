"""Tests for the loop engine using a mock provider."""

import tempfile
from pathlib import Path

import pytest

from atomics.config import AtomicsSettings
from atomics.core.engine import LoopEngine
from atomics.models import BurnTier
from atomics.providers.base import BaseProvider, ProviderResponse
from atomics.storage.repository import MetricsRepository


class MockProvider(BaseProvider):
    def __init__(self) -> None:
        self.call_count = 0

    @property
    def name(self) -> str:
        return "mock"

    async def generate(self, prompt, *, system="", model=None, max_tokens=1024):
        self.call_count += 1
        return ProviderResponse(
            text=f"response #{self.call_count}",
            input_tokens=30,
            output_tokens=60,
            total_tokens=90,
            model=model or "mock-model",
            latency_ms=10.0,
            estimated_cost_usd=0.0005,
        )

    async def health_check(self):
        return True


def _make_engine(tier=BurnTier.EZ, interval=0, budget=10.0):
    settings = AtomicsSettings(anthropic_api_key="fake")
    provider = MockProvider()
    db_path = Path(tempfile.mktemp(suffix=".db"))
    repo = MetricsRepository(db_path)
    engine = LoopEngine(
        provider=provider,
        repo=repo,
        settings=settings,
        tier=tier,
        interval_override=interval,
        budget_override=budget,
    )
    return engine, provider, repo


@pytest.mark.asyncio
async def test_engine_runs_n_iterations():
    engine, provider, repo = _make_engine()
    await engine.run(max_iterations=5)
    assert provider.call_count == 5
    runs = repo.get_recent_runs(limit=1)
    assert runs[0]["total_tasks"] == 5
    assert runs[0]["successful_tasks"] == 5
    repo.close()


@pytest.mark.asyncio
async def test_engine_respects_tier():
    engine, _, repo = _make_engine(tier=BurnTier.MEGA)
    assert engine.tier == BurnTier.MEGA
    await engine.run(max_iterations=1)
    repo.close()


@pytest.mark.asyncio
async def test_engine_records_metrics():
    engine, _, repo = _make_engine()
    await engine.run(max_iterations=3)
    runs = repo.get_recent_runs(limit=1)
    assert runs[0]["total_tokens"] == 270  # 90 * 3
    assert runs[0]["total_cost_usd"] == pytest.approx(0.0015, abs=0.0001)
    repo.close()


@pytest.mark.asyncio
async def test_engine_stops_on_budget():
    engine, provider, repo = _make_engine(budget=0.0008)
    await engine.run(max_iterations=100)
    assert provider.call_count < 100
    repo.close()
