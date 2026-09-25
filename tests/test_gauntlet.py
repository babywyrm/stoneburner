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


def test_run_spread_is_the_stdev_of_per_run_means() -> None:
    from types import SimpleNamespace

    from atomics.eval.gauntlet import run_mean_stdev

    summary = SimpleNamespace(
        runs=3,
        results=[
            SimpleNamespace(run_scores=[0.8, 0.9, 1.0]),
            SimpleNamespace(run_scores=[0.4, 0.5, 0.6]),
        ],
    )
    assert run_mean_stdev(summary) == pytest.approx(0.1)

    uneven = SimpleNamespace(
        runs=3,
        results=[
            SimpleNamespace(run_scores=[0.8, 0.9]),
            SimpleNamespace(run_scores=[0.4, 0.5, 0.6]),
        ],
    )
    assert run_mean_stdev(uneven) is None
    assert run_mean_stdev(SimpleNamespace(runs=1, results=summary.results)) is None


def test_spread_shows_in_the_log_and_the_table() -> None:
    from atomics.eval.gauntlet import format_job_log

    row = SuiteJobResult(model="m", suite="redblue", ok=True, headline=0.806, stdev=0.012)
    assert format_job_log(row) == "ok m redblue headline=0.806 stdev=0.012"
    assert format_headline_cell(row) == "80.6% ±1.2"


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
async def test_resume_skips_finished_jobs_and_reruns_the_rest(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    status.write_text(
        json.dumps(
            {
                "started_at": "2026-09-22T04:05:12+00:00",
                "models": ["done:1b", "cut:30b"],
                "suites": ["redblue", "toolcall"],
                "current_model": "cut:30b",
                "current_suite": "redblue",
                "completed": [
                    {"model": "done:1b", "suite": "redblue", "ok": True, "headline": 0.9,
                     "error": None, "tool_capable": None, "exit_code": 0},
                    {"model": "done:1b", "suite": "toolcall", "ok": False, "headline": None,
                     "error": "model did not emit a tool call", "tool_capable": False,
                     "exit_code": 1},
                ],
                "finished_at": None,
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, str]] = []

    async def run_suite(*, model: str, suite: str, skip_incapable: bool) -> SuiteJobResult:
        calls.append((model, suite))
        return SuiteJobResult(model=model, suite=suite, ok=True, headline=0.5)

    results = await run_gauntlet(
        models=["done:1b", "cut:30b"],
        suites=["redblue", "toolcall"],
        run_suite=run_suite,
        status_path=status,
        resume=True,
    )

    assert calls == [("done:1b", "toolcall"), ("cut:30b", "redblue"), ("cut:30b", "toolcall")]
    assert [(r.model, r.suite, r.headline) for r in results] == [
        ("done:1b", "redblue", 0.9),
        ("done:1b", "toolcall", 0.5),
        ("cut:30b", "redblue", 0.5),
        ("cut:30b", "toolcall", 0.5),
    ]
    final = json.loads(status.read_text(encoding="utf-8"))
    assert final["started_at"] == "2026-09-22T04:05:12+00:00"
    assert [(c["model"], c["suite"]) for c in final["completed"]] == [
        ("done:1b", "redblue"),
        ("done:1b", "toolcall"),
        ("cut:30b", "redblue"),
        ("cut:30b", "toolcall"),
    ]
    assert final["finished_at"]


@pytest.mark.asyncio
async def test_sweep_save_writes_suite_rows_and_closes_the_parent(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock

    from atomics.commands.sweep import suite_persister
    from atomics.eval.codereview import run_codereview
    from atomics.eval.redblue.fixtures import RED_FIXTURES
    from atomics.eval.redblue.runner import run_redblue
    from atomics.providers.base import ProviderResponse
    from atomics.storage import MetricsRepository
    from tests.test_redblue import _judge, _provider

    def _reply(text: str) -> ProviderResponse:
        return ProviderResponse(
            text=text,
            input_tokens=5,
            output_tokens=5,
            total_tokens=10,
            model="m",
            latency_ms=1.0,
            estimated_cost_usd=0.0,
        )

    reviewer = AsyncMock()
    reviewer.name = "ollama"
    reviewer.generate = AsyncMock(return_value=_reply("SQL injection. Use bound parameters."))
    grader = AsyncMock()
    grader.name = "judge"
    grader.generate = AsyncMock(return_value=_reply("VERDICT: DETECTED\nRATIONALE: ok"))

    redblue = await run_redblue(_provider(), judge_provider=_judge(), mode="red")
    review = await run_codereview(reviewer, judge_provider=grader, model="m:1b")
    db = tmp_path / "atomics.db"
    persist = suite_persister(db)
    persist("redblue", "ollama", "m:1b", redblue)
    persist("codereview", "ollama", "m:1b", review)

    repo = MetricsRepository(db)
    try:
        assert len(repo.get_run_tasks(redblue.run_id)) == len(RED_FIXTURES)
        assert len(repo.get_evaluation_results(run_id=review.run_id)) == len(review.results)
        assert repo.get_run(redblue.run_id)["completed_at"]
        assert repo.get_run(review.run_id)["completed_at"]
    finally:
        repo.close()


@pytest.mark.parametrize(("flag", "saves"), [("--save", True), ("--no-save", False)])
def test_sweep_save_reaches_the_suite_runner(monkeypatch, flag: str, saves: bool) -> None:
    from types import SimpleNamespace

    from click.testing import CliRunner

    from atomics.cli import cli

    seen: dict[str, object] = {}

    def fake_runner(**kwargs):
        seen.update(kwargs)
        return None

    async def fake_gauntlet(**_kwargs):
        return [SuiteJobResult(model="a:1b", suite="redblue", ok=True, headline=0.9)]

    monkeypatch.setattr("atomics.eval.gauntlet.make_suite_runner", fake_runner)
    monkeypatch.setattr("atomics.eval.gauntlet.run_gauntlet", fake_gauntlet)
    monkeypatch.setattr(
        "atomics.commands.sweep._make_provider",
        lambda *_a, **_k: SimpleNamespace(name="ollama", default_model="m"),
    )
    result = CliRunner().invoke(cli, ["sweep", "--models", "a:1b", "--suites", "redblue", flag])

    assert result.exit_code == 0, result.output
    assert (seen["persist"] is not None) is saves


def test_sweep_resume_needs_a_status_file() -> None:
    from click.testing import CliRunner

    from atomics.cli import cli

    result = CliRunner().invoke(
        cli, ["sweep", "--models", "a:1b", "--suites", "redblue", "--resume"]
    )

    assert result.exit_code != 0
    assert "--resume needs --status" in result.output


def test_sweep_resume_is_forwarded(monkeypatch, tmp_path: Path) -> None:
    from types import SimpleNamespace

    from click.testing import CliRunner

    from atomics.cli import cli

    seen: dict[str, object] = {}

    async def fake_gauntlet(**kwargs):
        seen.update(kwargs)
        return [SuiteJobResult(model="a:1b", suite="redblue", ok=True, headline=0.9)]

    monkeypatch.setattr("atomics.eval.gauntlet.run_gauntlet", fake_gauntlet)
    monkeypatch.setattr(
        "atomics.commands.sweep._make_provider",
        lambda *_a, **_k: SimpleNamespace(name="ollama", default_model="m"),
    )
    result = CliRunner().invoke(
        cli,
        [
            "sweep",
            "--models",
            "a:1b",
            "--suites",
            "redblue",
            "--status",
            str(tmp_path / "s.json"),
            "--resume",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["resume"] is True


@pytest.mark.asyncio
async def test_resume_without_a_status_file_starts_fresh(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    calls: list[tuple[str, str]] = []

    async def run_suite(*, model: str, suite: str, skip_incapable: bool) -> SuiteJobResult:
        calls.append((model, suite))
        return SuiteJobResult(model=model, suite=suite, ok=True, headline=0.5)

    await run_gauntlet(
        models=["a:1b"], suites=["redblue"], run_suite=run_suite, status_path=status, resume=True
    )

    assert calls == [("a:1b", "redblue")]


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
async def test_suite_runner_hands_each_summary_to_persist(monkeypatch) -> None:
    from types import SimpleNamespace

    summary = SimpleNamespace(calibration_score=0.9)

    async def fake_refusal(*_args, **_kwargs):
        return summary

    monkeypatch.setattr("atomics.eval.refusal.run_refusal", fake_refusal)
    saved: list[tuple[str, str, str, object]] = []
    run_suite = make_suite_runner(
        provider_factory=lambda _model: SimpleNamespace(name="ollama"),
        judge_provider=SimpleNamespace(name="ollama"),
        judge_model="judge",
        runs=1,
        thinking=False,
        thinking_budget=None,
        persist=lambda *args: saved.append(args),
    )

    result = await run_suite(model="m:1b", suite="refusal", skip_incapable=False)

    assert result.ok is True
    assert saved == [("refusal", "ollama", "m:1b", summary)]


@pytest.mark.asyncio
async def test_a_failed_save_fails_the_job(monkeypatch) -> None:
    from types import SimpleNamespace

    async def fake_refusal(*_args, **_kwargs):
        return SimpleNamespace(calibration_score=0.9)

    def broken(*_args):
        raise RuntimeError("database is locked")

    monkeypatch.setattr("atomics.eval.refusal.run_refusal", fake_refusal)
    run_suite = make_suite_runner(
        provider_factory=lambda _model: SimpleNamespace(name="ollama"),
        judge_provider=SimpleNamespace(name="ollama"),
        judge_model="judge",
        runs=1,
        thinking=False,
        thinking_budget=None,
        persist=broken,
    )

    result = await run_suite(model="m:1b", suite="refusal", skip_incapable=False)

    assert result.ok is False
    assert result.headline == 0.9
    assert result.error is not None and result.error.startswith("save failed")


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
