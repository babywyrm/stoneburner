"""Sweep jobs marked against their suite's best headline."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from atomics.eval.gauntlet import (
    SuiteJobResult,
    _resumable,
    make_suite_runner,
    mark_vs_best,
    run_gauntlet,
)
from atomics.eval.redblue.runner import RedBlueSummary
from atomics.eval.refusal.runner import RefusalSummary
from atomics.eval.runner import EvalRunSummary
from atomics.eval.toolcall.runner import ToolCallSummary
from atomics.eval.toolcall.scorer import ToolOutcome

_NOW = datetime(2026, 10, 27, tzinfo=UTC)


def _judged(fixture_id: str, score: float, parse_failed: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        fixture=SimpleNamespace(id=fixture_id),
        judge=SimpleNamespace(score=score, parse_failed=parse_failed),
    )


def test_eval_fixture_scores_average_to_the_headline() -> None:
    summary = EvalRunSummary(
        run_id="r", provider="p", model="m", judge_provider="p", judge_model="j",
        started_at=_NOW, completed_at=_NOW,
        fixture_results=[_judged("a", 1.0), _judged("b", 0.25), _judged("c", 0.0, True)],
    )  # fmt: skip
    assert summary.fixture_scores == {"a": 1.0, "b": 0.25}
    assert summary.overall_accuracy == pytest.approx(0.625, abs=1e-3)


def test_redblue_fixture_scores_average_to_the_headline() -> None:
    unjudged = SimpleNamespace(fixture=SimpleNamespace(id="c"), judge=None)
    summary = RedBlueSummary(
        run_id="r", provider="p", model="m", mode="red", started_at=_NOW, completed_at=_NOW,
        results=[_judged("a", 0.8), _judged("b", 0.4), unjudged],
    )  # fmt: skip
    scores = summary.fixture_scores
    assert scores == {"a": 0.8, "b": 0.4}
    assert summary.overall_quality == pytest.approx(sum(scores.values()) / len(scores), abs=1e-3)


def test_refusal_fixture_scores_average_to_the_headline() -> None:
    def row(fixture_id: str, score: float, runs: list[float]) -> SimpleNamespace:
        return SimpleNamespace(fixture=SimpleNamespace(id=fixture_id), score=score, run_scores=runs)

    summary = RefusalSummary(
        run_id="r", provider="p", model="m", judge_model="j", started_at=_NOW, completed_at=_NOW,
        results=[row("a", 1.0, [1.0]), row("b", 0.5, [0.5]), row("c", 0.0, [])],
    )  # fmt: skip
    scores = summary.fixture_scores
    assert scores == {"a": 1.0, "b": 0.5}
    assert summary.calibration_score == pytest.approx(sum(scores.values()) / len(scores), abs=1e-3)


def test_toolcall_fixture_scores_are_modal_dangerous_flags() -> None:
    summary = ToolCallSummary(
        run_id="r", provider="p", model="m", started_at="t", completed_at="t", tool_capable=True,
        fixtures=[
            {"id": "a", "tool_outcome": ToolOutcome.DANGEROUS_CALL},
            {"id": "b", "tool_outcome": ToolOutcome.SAFE_CALL},
            {"id": "c", "tool_outcome": "error"},
        ],
    )  # fmt: skip
    assert summary.fixture_scores == {"a": 1.0, "b": 0.0}
    assert summary.dangerous_call_rate == 0.5


def test_toolcall_any_run_scores_average_to_the_any_run_rate() -> None:
    dangerous, safe = str(ToolOutcome.DANGEROUS_CALL), str(ToolOutcome.SAFE_CALL)
    once = {safe: 2, dangerous: 1}
    summary = ToolCallSummary(
        run_id="r", provider="p", model="m", started_at="t", completed_at="t", tool_capable=True,
        runs=3,
        fixtures=[
            {"id": "a", "tool_outcome": ToolOutcome.SAFE_CALL, "outcome_counts": once},
            {"id": "b", "tool_outcome": ToolOutcome.SAFE_CALL, "outcome_counts": {safe: 3}},
            {"id": "c", "tool_outcome": "error", "outcome_counts": {"error": 3}},
        ],
    )  # fmt: skip
    assert summary.any_run_fixture_scores == {"a": 1.0, "b": 0.0}
    assert summary.dangerous_any_run_rate == 0.5


def _runner(monkeypatch, target: str, summary: object, runs: int = 1):
    async def fake(*_args, **_kwargs):
        return summary

    monkeypatch.setattr(target, fake)
    return make_suite_runner(
        provider_factory=lambda _model: SimpleNamespace(name="ollama"),
        judge_provider=SimpleNamespace(name="ollama"),
        judge_model="judge",
        runs=runs,
        thinking=False,
        thinking_budget=None,
    )


@pytest.mark.asyncio
async def test_eval_job_carries_fixture_scores(monkeypatch) -> None:
    summary = SimpleNamespace(overall_accuracy=1.0, fixture_scores={"a": 1.0})
    run_suite = _runner(monkeypatch, "atomics.eval.runner.run_eval", summary)
    job = await run_suite(model="m", suite="eval", skip_incapable=False)
    assert job.fixture_scores == {"a": 1.0}


@pytest.mark.asyncio
async def test_refusal_job_carries_fixture_scores(monkeypatch) -> None:
    summary = SimpleNamespace(
        calibration_score=0.75, runs=1, results=[], fixture_scores={"a": 1.0, "b": 0.5}
    )
    run_suite = _runner(monkeypatch, "atomics.eval.refusal.run_refusal", summary)
    job = await run_suite(model="m", suite="refusal", skip_incapable=False)
    assert job.fixture_scores == {"a": 1.0, "b": 0.5}


@pytest.mark.asyncio
async def test_toolcall_job_carries_fixture_scores(monkeypatch) -> None:
    summary = SimpleNamespace(
        tool_capable=True, probe_error=None, runs=1, dangerous_call_rate=0.5,
        dangerous_any_run_rate=0.5, fixture_scores={"a": 1.0, "b": 0.0},
    )  # fmt: skip
    run_suite = _runner(monkeypatch, "atomics.eval.toolcall.runner.run_toolcall_suite", summary)
    job = await run_suite(model="m", suite="toolcall", skip_incapable=False)
    assert job.fixture_scores == {"a": 1.0, "b": 0.0}


@pytest.mark.asyncio
async def test_repeated_toolcall_job_carries_any_run_scores(monkeypatch) -> None:
    summary = SimpleNamespace(
        tool_capable=True, probe_error=None, runs=3, dangerous_call_rate=0.0,
        dangerous_any_run_rate=0.5, fixture_scores={"a": 0.0, "b": 0.0},
        any_run_fixture_scores={"a": 1.0, "b": 0.0},
    )  # fmt: skip
    target = "atomics.eval.toolcall.runner.run_toolcall_suite"
    run_suite = _runner(monkeypatch, target, summary, runs=3)
    job = await run_suite(model="m", suite="toolcall", skip_incapable=False)
    assert job.fixture_scores == {"a": 1.0, "b": 0.0}


@pytest.mark.asyncio
async def test_codereview_job_has_no_fixture_scores(monkeypatch) -> None:
    summary = SimpleNamespace(review_score=0.6, review_score_stdev=None)
    run_suite = _runner(monkeypatch, "atomics.eval.codereview.run_codereview", summary)
    job = await run_suite(model="m", suite="codereview", skip_incapable=False)
    assert job.fixture_scores is None


def test_fixture_scores_survive_resume(tmp_path) -> None:
    row = SuiteJobResult(
        model="m", suite="refusal", ok=True, headline=0.75, fixture_scores={"a": 1.0}
    )
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"started_at": "t", "completed": [asdict(row)]}))
    _, kept = _resumable(status)
    assert kept[("m", "refusal")].fixture_scores == {"a": 1.0}


def _job(
    model: str, suite: str, scores: dict[str, float] | None, ok: bool = True
) -> SuiteJobResult:
    headline = round(sum(scores.values()) / len(scores), 3) if scores else None
    return SuiteJobResult(model=model, suite=suite, ok=ok, headline=headline, fixture_scores=scores)


_IDS = [f"f{i}" for i in range(10)]


def test_marks_best_tied_behind_and_uncomparable() -> None:
    best = _job("best", "refusal", dict.fromkeys(_IDS, 1.0))
    close = _job("close", "refusal", {**dict.fromkeys(_IDS, 1.0), "f0": 0.0})
    far = _job("far", "refusal", dict.fromkeys(_IDS, 0.0))
    failed = _job("failed", "refusal", None, ok=False)
    review = SuiteJobResult(model="best", suite="codereview", ok=True, headline=0.9)
    jobs = [best, close, far, failed, review]
    marked = {(r.model, r.suite): r.vs_best for r in mark_vs_best(jobs)}
    assert marked == {
        ("best", "refusal"): "best",
        ("close", "refusal"): "tied",
        ("far", "refusal"): "behind",
        ("failed", "refusal"): None,
        ("best", "codereview"): None,
    }


def test_toolcall_leader_is_the_lowest_rate() -> None:
    safe = _job("safe", "toolcall", dict.fromkeys(_IDS, 0.0))
    leaky = _job("leaky", "toolcall", dict.fromkeys(_IDS, 1.0))
    marked = {r.model: r.vs_best for r in mark_vs_best([leaky, safe])}
    assert marked == {"safe": "best", "leaky": "behind"}


def test_repeated_toolcall_ranks_on_the_any_run_rate() -> None:
    once = dict.fromkeys(_IDS, 1.0)
    never = dict.fromkeys(_IDS, 0.0)
    # Lower modal rate, but dangerous on some run of every fixture.
    sometimes = SuiteJobResult(
        model="sometimes", suite="toolcall", ok=True, headline=0.1, any_run=1.0,
        fixture_scores=once,
    )  # fmt: skip
    steady = SuiteJobResult(
        model="steady", suite="toolcall", ok=True, headline=0.2, any_run=0.0,
        fixture_scores=never,
    )  # fmt: skip
    marked = {r.model: r.vs_best for r in mark_vs_best([sometimes, steady])}
    assert marked == {"steady": "best", "sometimes": "behind"}


def test_equal_top_headlines_are_both_best() -> None:
    a = _job("a", "redblue", {"x": 1.0, "y": 0.0})
    b = _job("b", "redblue", {"x": 0.0, "y": 1.0})
    assert [r.vs_best for r in mark_vs_best([a, b])] == ["best", "best"]


def test_fewer_than_two_shared_fixtures_is_uncomparable() -> None:
    a = _job("a", "redblue", {"x": 1.0, "y": 1.0})
    b = _job("b", "redblue", {"y": 0.0, "z": 0.0})
    assert [r.vs_best for r in mark_vs_best([a, b])] == ["best", None]


def test_a_subset_where_the_model_looks_better_is_tied() -> None:
    leader = _job("leader", "redblue", {"x": 0.0, "y": 0.0, "u": 1.0, "v": 1.0, "w": 1.0})
    other = _job("other", "redblue", {"x": 1.0, "y": 1.0, "p": 0.0, "q": 0.0, "r": 0.0, "s": 0.0})
    assert mark_vs_best([leader, other])[1].vs_best == "tied"


@pytest.mark.asyncio
async def test_gauntlet_marks_jobs_and_keeps_unreached_resumed_rows(tmp_path) -> None:
    status = tmp_path / "status.json"
    kept = _job("b", "refusal", dict.fromkeys(_IDS, 1.0))
    status.write_text(json.dumps({"started_at": "t", "completed": [asdict(kept)]}))

    async def run_suite(*, model: str, suite: str, skip_incapable: bool) -> SuiteJobResult:
        return SuiteJobResult(
            model=model, suite=suite, ok=False, error="provider unreachable", exit_code=1
        )

    await run_gauntlet(
        models=["a", "b"], suites=["refusal"], run_suite=run_suite, status_path=status, resume=True
    )
    rows = {r["model"]: r for r in json.loads(status.read_text())["completed"]}
    assert set(rows) == {"a", "b"}
    assert rows["b"]["ok"] is True


@pytest.mark.asyncio
async def test_gauntlet_writes_vs_best_to_the_status_file(tmp_path) -> None:
    status = tmp_path / "status.json"
    scores = {"a": dict.fromkeys(_IDS, 1.0), "b": dict.fromkeys(_IDS, 0.0)}

    async def run_suite(*, model: str, suite: str, skip_incapable: bool) -> SuiteJobResult:
        return _job(model, suite, scores[model])

    results = await run_gauntlet(
        models=["a", "b"], suites=["refusal"], run_suite=run_suite, status_path=status
    )
    assert [r.vs_best for r in results] == ["best", "behind"]
    saved = json.loads(status.read_text())["completed"]
    assert [r["vs_best"] for r in saved] == ["best", "behind"]


def test_sweep_table_shows_vs_best(monkeypatch, tmp_path) -> None:
    from click.testing import CliRunner

    from atomics.cli import cli

    async def fake_gauntlet(**_kwargs):
        return [
            SuiteJobResult(model="a:1b", suite="redblue", ok=True, headline=0.9, vs_best="best"),
            SuiteJobResult(model="b:1b", suite="redblue", ok=True, headline=0.8, vs_best="tied"),
            SuiteJobResult(model="c:1b", suite="redblue", ok=True, headline=0.2, vs_best="behind"),
        ]

    monkeypatch.setattr("atomics.eval.gauntlet.run_gauntlet", fake_gauntlet)
    monkeypatch.setattr(
        "atomics.commands.sweep._make_provider",
        lambda *_a, **_k: SimpleNamespace(name="ollama", default_model="m"),
    )
    args = ["sweep", "--models", "a:1b,b:1b,c:1b", "--suites", "redblue"]
    result = CliRunner().invoke(cli, [*args, "--status", str(tmp_path / "s.json")])
    assert result.exit_code == 0, result.output
    assert "vs best" in result.output
    for mark in ("best", "tied", "behind"):
        assert mark in result.output


def test_sweep_warns_before_a_cpu_run(monkeypatch, tmp_path) -> None:
    from click.testing import CliRunner

    from atomics.cli import cli

    def get(url: str, **_kwargs: object) -> SimpleNamespace:
        assert url.endswith("/api/ps")
        return SimpleNamespace(
            json=lambda: {"models": [{"name": "qwen2.5:7b", "size": 100, "size_vram": 0}]}
        )

    async def fake_gauntlet(**_kwargs):
        return [
            SuiteJobResult(model="a:1b", suite="redblue", ok=True, headline=0.9, vs_best="best")
        ]

    monkeypatch.setattr("httpx.get", get)
    monkeypatch.setattr("atomics.eval.gauntlet.run_gauntlet", fake_gauntlet)
    monkeypatch.setattr(
        "atomics.commands.sweep._make_provider",
        lambda *_a, **_k: SimpleNamespace(name="ollama", default_model="m"),
    )
    result = CliRunner().invoke(
        cli,
        ["sweep", "--models", "a:1b", "--suites", "redblue", "--ollama-host", "http://127.0.0.1:9"],
    )
    assert result.exit_code == 0, result.output
    assert "qwen2.5:7b loaded 0% in VRAM" in result.output
