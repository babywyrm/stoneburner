"""Tests for the multi-suite overnight sweep driver."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atomics.eval.gauntlet import (
    SuiteJobResult,
    format_headline_cell,
    make_suite_runner,
    parse_suites,
    run_gauntlet,
)


def test_headline_cell_names_a_toolcall_rate() -> None:
    assert (
        format_headline_cell(
            SuiteJobResult(model="m", suite="toolcall", ok=True, headline=0.75)
        )
        == "dangerous 75.0%"
    )
    assert (
        format_headline_cell(
            SuiteJobResult(model="m", suite="redblue", ok=True, headline=0.0)
        )
        == "0.0%"
    )
    assert (
        format_headline_cell(
            SuiteJobResult(model="m", suite="toolcall", ok=False, headline=None)
        )
        == "—"
    )


def test_parse_suites_preserves_order_and_dedupes() -> None:
    assert parse_suites("redblue,refusal,toolcall,codereview") == [
        "redblue",
        "refusal",
        "toolcall",
        "codereview",
    ]
    assert parse_suites("eval, eval, redblue") == ["eval", "redblue"]


def test_parse_suites_rejects_unknown_and_empty() -> None:
    with pytest.raises(ValueError, match="unknown"):
        parse_suites("redblue,nope")
    with pytest.raises(ValueError, match="empty"):
        parse_suites("  , ")


@pytest.mark.asyncio
async def test_gauntlet_runs_each_model_suite_and_defaults_skip_incapable_off() -> None:
    calls: list[tuple[str, str, bool]] = []

    async def run_suite(*, model: str, suite: str, skip_incapable: bool) -> SuiteJobResult:
        calls.append((model, suite, skip_incapable))
        return SuiteJobResult(model=model, suite=suite, ok=True, headline=0.9)

    results = await run_gauntlet(
        models=["granite4.1:8b", "qwen3:14b"],
        suites=["redblue", "toolcall"],
        run_suite=run_suite,
    )

    assert calls == [
        ("granite4.1:8b", "redblue", False),
        ("granite4.1:8b", "toolcall", False),
        ("qwen3:14b", "redblue", False),
        ("qwen3:14b", "toolcall", False),
    ]
    assert [r.suite for r in results] == ["redblue", "toolcall", "redblue", "toolcall"]


@pytest.mark.asyncio
async def test_gauntlet_continues_after_a_suite_failure() -> None:
    async def run_suite(*, model: str, suite: str, skip_incapable: bool) -> SuiteJobResult:
        if suite == "toolcall":
            return SuiteJobResult(
                model=model,
                suite=suite,
                ok=False,
                tool_capable=False,
                error="not tool capable",
                exit_code=1,
            )
        return SuiteJobResult(model=model, suite=suite, ok=True, headline=0.8)

    results = await run_gauntlet(
        models=["tiny:1b"],
        suites=["redblue", "toolcall", "refusal"],
        run_suite=run_suite,
    )

    assert [r.ok for r in results] == [True, False, True]
    assert results[1].exit_code == 1


@pytest.mark.asyncio
async def test_gauntlet_rewrites_status_and_appends_log(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    log = tmp_path / "sweep.log"

    async def run_suite(*, model: str, suite: str, skip_incapable: bool) -> SuiteJobResult:
        payload = json.loads(status.read_text(encoding="utf-8"))
        assert payload["current_model"] == model
        assert payload["current_suite"] == suite
        assert payload["finished_at"] is None
        return SuiteJobResult(model=model, suite=suite, ok=True, headline=0.97)

    await run_gauntlet(
        models=["granite4.1:8b"],
        suites=["redblue"],
        run_suite=run_suite,
        status_path=status,
        log_path=log,
    )

    final = json.loads(status.read_text(encoding="utf-8"))
    assert final["finished_at"]
    assert final["current_model"] is None
    assert final["completed"][0]["headline"] == 0.97
    text = log.read_text(encoding="utf-8")
    assert "granite4.1:8b" in text
    assert "redblue" in text


@pytest.mark.asyncio
async def test_gauntlet_log_names_headline_and_failure(tmp_path: Path) -> None:
    log = tmp_path / "sweep.log"

    async def run_suite(*, model: str, suite: str, skip_incapable: bool) -> SuiteJobResult:
        if model == "scored" and suite == "redblue":
            return SuiteJobResult(model=model, suite=suite, ok=True, headline=0.0)
        if model == "scored" and suite == "toolcall":
            return SuiteJobResult(
                model=model,
                suite=suite,
                ok=True,
                headline=0.75,
                tool_capable=True,
            )
        if suite == "toolcall":
            return SuiteJobResult(
                model=model,
                suite=suite,
                ok=False,
                tool_capable=False,
                error="model did not emit a tool call",
                exit_code=1,
            )
        return SuiteJobResult(model=model, suite=suite, ok=True, headline=0.5)

    await run_gauntlet(
        models=["scored", "tiny:1b"],
        suites=["redblue", "toolcall"],
        run_suite=run_suite,
        log_path=log,
    )

    text = log.read_text(encoding="utf-8")
    assert "ok scored redblue headline=0.000" in text
    assert "ok scored toolcall dangerous_call_rate=0.750" in text
    assert "fail tiny:1b toolcall model did not emit a tool call" in text


@pytest.mark.asyncio
async def test_suite_runner_marks_toolcall_incapable_as_failure(monkeypatch) -> None:
    from types import SimpleNamespace

    async def fake_toolcall(**_kwargs):
        return SimpleNamespace(tool_capable=False, dangerous_call_rate=None)

    monkeypatch.setattr(
        "atomics.eval.toolcall.runner.run_toolcall_suite",
        fake_toolcall,
    )
    run_suite = make_suite_runner(
        provider_factory=lambda _model: SimpleNamespace(name="ollama"),
        judge_provider=SimpleNamespace(name="ollama"),
        judge_model="judge",
        runs=3,
        thinking=False,
        thinking_budget=8000,
    )

    result = await run_suite(
        model="tiny:1b",
        suite="toolcall",
        skip_incapable=False,
    )

    assert result.ok is False
    assert result.tool_capable is False
    assert result.exit_code == 1
