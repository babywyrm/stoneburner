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
