"""Shared pytest fixtures and test doubles for the Atomics test suite."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from atomics.config import AtomicsSettings
from atomics.core.engine import LoopEngine
from atomics.models import BurnTier
from atomics.providers.base import BaseProvider, ProviderResponse
from atomics.storage.repository import MetricsRepository


class MockProvider(BaseProvider):
    """Minimal provider that returns deterministic token counts."""

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


@pytest.fixture
def atomics_settings() -> AtomicsSettings:
    return AtomicsSettings(anthropic_api_key="fake-key-for-tests")


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "metrics.sqlite"


@pytest.fixture
def metrics_repo(tmp_db_path: Path) -> MetricsRepository:
    return MetricsRepository(tmp_db_path)


@pytest.fixture
def make_engine(
    atomics_settings: AtomicsSettings,
    metrics_repo: MetricsRepository,
) -> Callable[..., tuple[LoopEngine, MockProvider, MetricsRepository]]:
    """Build a LoopEngine with MockProvider on the shared tmp DB."""

    def _make(
        *,
        tier: BurnTier = BurnTier.EZ,
        interval: int | None = 0,
        budget: float | None = 10.0,
    ) -> tuple[LoopEngine, MockProvider, MetricsRepository]:
        provider = MockProvider()
        engine = LoopEngine(
            provider=provider,
            repo=metrics_repo,
            settings=atomics_settings,
            tier=tier,
            interval_override=interval,
            budget_override=budget,
        )
        return engine, provider, metrics_repo

    return _make
