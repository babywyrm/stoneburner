"""Shared pytest fixtures and test doubles for the Atomics test suite."""

from __future__ import annotations

import asyncio
import socket
import sqlite3
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

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


# The Ollama, vLLM, and llama.cpp default ports. Even on loopback, a connection
# to one reaches a real inference server; test doubles bind ephemeral ports.
_INFERENCE_PORTS = frozenset({11434, 8000, 8080})
_LOOPBACK = frozenset({"127.0.0.1", "::1"})
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


@pytest.fixture(autouse=True)
def _sweep_skips_the_vram_probe(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep asks Ollama ``/api/ps`` before it starts. That port is guarded.

    ``test_doctor`` and ``test_sweep_warns_before_a_cpu_run`` call the real
    probe against a mocked ``httpx.get``.
    """
    nodeid = request.node.nodeid
    if "test_doctor.py" in nodeid or "test_sweep_warns_before_a_cpu_run" in nodeid:
        return
    monkeypatch.setattr("atomics.doctor._warn_cpu_offload", lambda *_a, **_k: None)


@pytest.fixture(autouse=True)
def _no_network(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[list[str]]:
    """Refuse and record any connection past loopback or to an inference port.

    The test fails at teardown even when the code under test swallowed the
    refusal, as health checks do. Mark a live test `network` to opt out.
    """
    attempts: list[str] = []
    if request.node.get_closest_marker("network"):
        yield attempts
        return

    def _guard(address: object) -> None:
        if isinstance(address, tuple) and len(address) >= 2:
            host, port = str(address[0]), address[1]
            if host not in _LOOPBACK or port in _INFERENCE_PORTS:
                attempts.append(f"{host}:{port}")
                raise ConnectionRefusedError(f"test network guard refused {host}:{port}")

    def connect(self: socket.socket, address: object) -> None:
        _guard(address)
        _real_connect(self, address)

    def connect_ex(self: socket.socket, address: object) -> int:
        _guard(address)
        return _real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    yield attempts
    if attempts:
        pytest.fail(f"test reached the network: {', '.join(attempts)}", pytrace=False)


@pytest.fixture
def jobs_never_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept submitted API jobs without running them; the work would reach a real provider."""

    async def _never_run(_self: object, _job: object, _work: object) -> None:
        return None

    monkeypatch.setattr("atomics.api.jobs.JobManager._run", _never_run)


@pytest.fixture
def offline_doctor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Doctor probes Ollama and AWS live; make both unreachable unless a test says otherwise."""
    import httpx

    def _unreachable(*_args: object, **_kwargs: object) -> None:
        raise httpx.ConnectError("offline in tests")

    monkeypatch.setattr(httpx, "get", _unreachable)
    try:
        import boto3
    except ImportError:
        return
    monkeypatch.setattr(boto3, "client", _unreachable)


@pytest.fixture(autouse=True)
def _isolated_operator_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings must not open the operator's database or read their keychain."""
    monkeypatch.setattr("atomics.paths.default_data_dir", lambda: tmp_path / "data")
    monkeypatch.setenv("ATOMICS_DB_PATH", str(tmp_path / "data" / "atomics.db"))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A --no-auth API server on loopback, lifespan started."""
    from fastapi.testclient import TestClient

    from atomics.api.config import ServerSettings
    from atomics.api.server import create_app

    with TestClient(create_app(ServerSettings(no_auth=True)), base_url="http://127.0.0.1") as tc:
        yield tc


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


class _LoopClock:
    """Stands in for the ``time`` module with ``monotonic`` read off an event loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def monotonic(self) -> float:
        return self._loop.time()

    def __getattr__(self, name: str) -> object:
        return getattr(time, name)


@pytest.fixture
async def fake_clock(monkeypatch: pytest.MonkeyPatch) -> Callable[[ModuleType], None]:
    """Run the test's event loop on virtual time.

    When nothing is runnable, the clock jumps to the next timer instead of
    waiting for it, so a 30-second soak takes no wall time and runs the same
    iterations on every host. A test double that only yields with
    ``asyncio.sleep(0)`` never lets the clock move: give it a nonzero sleep.
    Pass a module to make its ``time.monotonic()`` read the same clock.
    """
    loop = asyncio.get_running_loop()
    # asyncio has no public hook for its clock; SelectorEventLoop polls here.
    selector = loop._selector  # type: ignore[attr-defined]
    real_select = selector.select
    now = loop.time()

    def select(timeout: float | None = None) -> list[object]:
        nonlocal now
        if timeout is None:
            return real_select(None)
        events = real_select(0)
        if not events:
            now += timeout
        return events

    monkeypatch.setattr(loop, "time", lambda: now)
    monkeypatch.setattr(selector, "select", select)

    def _apply(module: ModuleType) -> None:
        monkeypatch.setattr(module, "time", _LoopClock(loop))

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
