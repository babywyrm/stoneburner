"""Async clients must close on the live loop, not during GC after asyncio.run."""

from __future__ import annotations

import asyncio

import pytest

from atomics.providers.base import BaseProvider


class _Dummy(BaseProvider):
    def __init__(self, client: object | None = None) -> None:
        self._client = client

    @property
    def name(self) -> str:
        return "dummy"

    async def generate(self, prompt: str, **kwargs: object) -> object:
        raise NotImplementedError

    async def health_check(self) -> bool:
        return True


class _AcloseClient:
    def __init__(self) -> None:
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


class _CloseClient:
    def __init__(self) -> None:
        self.closed = 0

    async def close(self) -> None:
        self.closed += 1


def test_aclose_prefers_aclose():
    client = _AcloseClient()
    asyncio.run(_Dummy(client).aclose())
    assert client.closed == 1


def test_aclose_falls_back_to_close():
    client = _CloseClient()
    asyncio.run(_Dummy(client).aclose())
    assert client.closed == 1


def test_aclose_providers_skips_none():
    from atomics.providers.base import aclose_providers

    client = _AcloseClient()
    asyncio.run(aclose_providers(_Dummy(client), None))
    assert client.closed == 1


def test_aclose_providers_skips_object_without_aclose():
    from atomics.providers.base import aclose_providers

    class _Bare:
        pass

    asyncio.run(aclose_providers(_Bare(), None))  # type: ignore[arg-type]


def test_guarded_provider_aclose_closes_inner():
    from atomics.eval.budget import EvalBudget, GuardedProvider

    client = _AcloseClient()
    inner = _Dummy(client)
    guarded = GuardedProvider(inner, EvalBudget().new_guard())
    asyncio.run(guarded.aclose())
    assert client.closed == 1


def test_run_async_closes_after_success():
    from atomics.commands.common import run_async

    client = _AcloseClient()
    p = _Dummy(client)

    async def work() -> int:
        assert client.closed == 0
        return 7

    assert run_async(work(), p) == 7
    assert client.closed == 1


def test_run_async_closes_after_failure():
    from atomics.commands.common import run_async

    client = _AcloseClient()
    p = _Dummy(client)

    async def work() -> int:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        run_async(work(), p)
    assert client.closed == 1


@pytest.mark.asyncio
async def test_api_and_worker_runners_aclose_providers(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock, patch

    from atomics.api import _discovery, _load, _runners, _sweep
    from atomics.api.models import (
        EvalRequest,
        ProviderTestRequest,
        RunRequest,
        SoakRequest,
        StressRequest,
        SweepRequest,
    )
    from atomics.distributed import worker_runner
    from atomics.distributed.models import TaskAssignment
    from atomics.eval.gauntlet import SuiteJobResult

    clients: list[_AcloseClient] = []

    def make_dummy(*_args: object, **_kwargs: object) -> _Dummy:
        c = _AcloseClient()
        clients.append(c)
        d = _Dummy(c)
        d.list_models = AsyncMock(return_value=[])  # type: ignore[attr-defined]
        d.generate = AsyncMock(  # type: ignore[method-assign]
            return_value=SimpleNamespace(
                text="4",
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                thinking_tokens=0,
                thinking_text="",
                latency_ms=1.0,
                estimated_cost_usd=0.0,
            )
        )
        return d

    monkeypatch.setattr(_discovery, "_provider", make_dummy)
    await _discovery.list_models("ollama", None)
    await _discovery.run_provider_test(ProviderTestRequest(provider="ollama"))
    assert [c.closed for c in clients] == [1, 1]

    clients.clear()
    monkeypatch.setattr(_runners, "_provider_for", make_dummy)
    with (
        patch.object(
            _runners,
            "run_eval",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(
                overall_accuracy=1.0, fixture_results=[], total_tokens=0, total_cost_usd=0.0
            ),
        ),
        patch.object(
            _runners,
            "run_rag",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(
                overall_rag_score=1.0, fixture_results=[], total_tokens=0, total_cost_usd=0.0
            ),
        ),
    ):
        await _runners.run_eval_from_request(EvalRequest(suite="accuracy", provider="ollama"))
        await _runners.run_eval_suite(EvalRequest(suite="rag", provider="ollama"))
    assert [c.closed for c in clients] == [1, 1, 1, 1]

    clients.clear()
    engine = MagicMock()
    engine.run = AsyncMock(
        return_value=SimpleNamespace(
            run_id="r",
            total_tasks=1,
            successful_tasks=1,
            failed_tasks=0,
            total_tokens=1,
            total_cost_usd=0.0,
        )
    )
    with (
        patch("atomics.core.engine.LoopEngine", return_value=engine),
        patch("atomics.storage.repository.MetricsRepository", return_value=MagicMock()),
        patch(
            "atomics.benchmark.tiers.get_tier_profile",
            return_value=SimpleNamespace(preferred_model=None),
        ),
    ):
        await _runners.run_benchmark_from_request(RunRequest(provider="ollama", iterations=1))
    assert [c.closed for c in clients] == [1]

    clients.clear()
    monkeypatch.setattr(_load, "_provider_for", make_dummy)
    with (
        patch.object(
            _load,
            "run_stress_provider",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(
                model="m",
                duration_seconds=1.0,
                total_tokens=0,
                total_requests=0,
                total_failed=0,
                peak_tps=1.0,
                saturation_concurrency=1,
                total_cost_usd=0.0,
                phases=[],
            ),
        ),
        patch.object(
            _load,
            "run_soak_provider",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(
                model="m",
                concurrency=1,
                duration_seconds=10,
                actual_duration_seconds=10.0,
                total_requests=0,
                total_failed=0,
                total_tokens=0,
                avg_tps=1.0,
                peak_tps=1.0,
                min_tps=1.0,
                avg_p95_ms=1.0,
                error_rate=0.0,
                throughput_drift_pct=0.0,
                latency_drift_pct=0.0,
                verdict="STABLE",
                total_cost_usd=0.0,
                samples=[],
            ),
        ),
    ):
        await _load.run_stress_from_request(
            StressRequest(provider="ollama", model="m", budget_usd=1.0)
        )
        await _load.run_soak_from_request(SoakRequest(provider="ollama", model="m", budget_usd=1.0))
    assert [c.closed for c in clients] == [1, 1]

    clients.clear()
    monkeypatch.setattr(_sweep, "_provider_for", make_dummy)
    with patch.object(
        _sweep,
        "_dispatch_suite" if hasattr(_sweep, "_dispatch_suite") else "run_gauntlet",
    ):
        pass
    with patch(
        "atomics.eval.gauntlet._dispatch_suite",
        new_callable=AsyncMock,
        return_value=(SuiteJobResult(model="a", suite="eval", ok=True, headline=1.0), None),
    ):
        await _sweep.run_sweep_from_request(
            SweepRequest(provider="ollama", models=["a"], suites=["eval"], budget_usd=1.0)
        )
    # 1 judge provider + 1 model provider built by provider_factory("a")
    assert [c.closed for c in clients] == [1, 1]

    clients.clear()
    monkeypatch.setattr(worker_runner, "make_provider", make_dummy)
    with patch.object(
        worker_runner,
        "execute_task",
        new_callable=AsyncMock,
        return_value={"status": "success"},
    ):
        await worker_runner.execute_assignment(
            TaskAssignment(
                assignment_id="a1",
                job_id="j1",
                worker_id="w1",
                task_spec={"task_name": "quick_question", "prompt": "hi"},
            )
        )
    assert [c.closed for c in clients] == [1]


def test_labcompare_cli_acloses_host_and_judge_providers(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch

    from click.testing import CliRunner

    from atomics.cli import cli
    from atomics.commands import labcompare as cmd_labcompare

    clients: list[_AcloseClient] = []

    def make_dummy(*_args: object, **_kwargs: object) -> _Dummy:
        c = _AcloseClient()
        clients.append(c)
        d = _Dummy(c)
        d._host = "http://127.0.0.1:11434"  # type: ignore[attr-defined]
        return d

    monkeypatch.setattr(cmd_labcompare, "_make_provider", make_dummy)
    with patch(
        "atomics.eval.runner.run_eval",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(overall_accuracy=0.9),
    ):
        res = CliRunner().invoke(
            cli,
            [
                "labcompare",
                "--host",
                "a=http://127.0.0.1:11434",
                "--host",
                "b=http://127.0.0.1:11435",
                "--models",
                "m",
                "--dimensions",
                "quality",
                "--quality-suite",
                "eval",
            ],
        )
    assert res.exit_code == 0, res.output
    assert [c.closed for c in clients] == [1, 1, 1, 1]


_PROVIDER_RUNNERS = (
    "run_eval",
    "run_rag",
    "run_codegen",
    "run_multiturn",
    "run_adversarial",
    "run_redblue",
    "run_refusal",
    "run_codereview",
    "run_toolcall_suite",
    "run_archreview",
    "run_probe",
    "run_agreement_study",
    "run_gauntlet",
    "run_model_sweep",
    "run_stress_provider",
    "run_soak_provider",
    "run_labcompare",
)


def test_command_modules_do_not_asyncio_run_provider_runners():
    """CLI must aclose HTTP clients on the live loop, not after asyncio.run.

    Battery `run` and sequential commands share one process. A leftover
    `asyncio.run(run_eval(...))` (and the other provider runners) is the
    Event-loop-is-closed traceback.
    """
    import re
    from pathlib import Path

    banned = re.compile(
        r"asyncio\.run\(\s*(?:" + "|".join(_PROVIDER_RUNNERS) + r")\b",
        re.MULTILINE,
    )
    commands = Path(__file__).resolve().parent.parent / "atomics" / "commands"
    leftovers: list[str] = []
    for path in commands.rglob("*.py"):
        text = path.read_text()
        for match in banned.finditer(text):
            line = text[: match.start()].count("\n") + 1
            leftovers.append(f"{path.relative_to(commands.parent)}:{line}")
    assert leftovers == []
