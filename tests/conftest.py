"""Shared pytest fixtures and test doubles for the Atomics test suite."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest

# Track and close all sqlite3 connections opened during tests so we don't leak
# them and trigger ResourceWarnings.
_orig_connect = sqlite3.connect
_open_connections: list[sqlite3.Connection] = []


def _tracked_connect(*args, **kwargs):  # type: ignore[no-untyped-def]
    conn = _orig_connect(*args, **kwargs)
    _open_connections.append(conn)
    return conn


sqlite3.connect = _tracked_connect

from atomics.config import AtomicsSettings
from atomics.core.engine import LoopEngine
from atomics.models import BurnTier
from atomics.providers.base import BaseProvider, ProviderResponse
from atomics.storage.repository import MetricsRepository
from tests.inference_stub import StubInferenceServer


class MockProvider(BaseProvider):
    """Minimal provider that returns deterministic token counts."""

    def __init__(self) -> None:
        self.call_count = 0

    @property
    def name(self) -> str:
        return "mock"

    async def generate(
        self,
        prompt,
        *,
        system="",
        model=None,
        max_tokens=1024,
        thinking=None,
        thinking_budget=None,
        temperature=None,
        effort=None,
        reasoning_mode=None,
    ):
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


@pytest.fixture(autouse=True)
def _close_sqlite_connections() -> Iterator[None]:
    """Close any sqlite3 connections opened during a test."""
    yield
    while _open_connections:
        conn = _open_connections.pop()
        try:
            conn.close()
        except Exception:
            pass


class _ScriptedClock:
    """Stands in for the ``time`` module with a scripted ``monotonic``.

    Returns 0.0 on the first call and ``elapsed`` on every call after, so a
    provider that brackets its HTTP call with two ``monotonic()`` reads
    measures exactly ``elapsed`` seconds. Any other attribute falls through to
    the real module.
    """

    def __init__(self, elapsed: float) -> None:
        self._elapsed = elapsed
        self._calls = 0

    def monotonic(self) -> float:
        tick = 0.0 if self._calls == 0 else self._elapsed
        self._calls += 1
        return tick

    def __getattr__(self, name: str) -> object:
        return getattr(time, name)


@pytest.fixture
def scripted_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[ModuleType, float], None]:
    """Pin a provider module's measured elapsed time to a known value.

    Providers time their own HTTP call with ``time.monotonic()``. Test doubles
    return in microseconds, so real elapsed time rounds to zero and throughput
    comes back undefined — making any assertion on latency or tokens/sec depend
    on how fast the machine is. Only the provider module's ``time`` reference is
    replaced, leaving the real module (which asyncio schedules against) intact.
    """

    def _apply(module: ModuleType, elapsed_seconds: float) -> None:
        monkeypatch.setattr(module, "time", _ScriptedClock(elapsed_seconds))

    return _apply


@pytest.fixture
def inference_stub() -> Iterator[StubInferenceServer]:
    """A real HTTP inference endpoint on an ephemeral port."""
    with StubInferenceServer() as stub:
        yield stub


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
