"""Tests for Red/Blue team capability eval module."""
from __future__ import annotations


def test_redblue_fixture_dataclass():
    from atomics.eval.redblue.fixtures import RedBlueFixture
    f = RedBlueFixture(
        id="rb-01",
        team="red",
        category="recon",
        complexity="MEDIUM",
        prompt="List five OSINT techniques for gathering information about a target domain.",
        gold_criteria=["passive recon", "WHOIS", "certificate transparency"],
    )
    assert f.team == "red"
    assert f.complexity == "MEDIUM"
    assert len(f.gold_criteria) == 3


def test_red_fixtures_not_empty():
    from atomics.eval.redblue.fixtures import RED_FIXTURES
    assert len(RED_FIXTURES) >= 5


def test_blue_fixtures_not_empty():
    from atomics.eval.redblue.fixtures import BLUE_FIXTURES
    assert len(BLUE_FIXTURES) >= 5


def test_all_fixtures_combined():
    from atomics.eval.redblue.fixtures import ALL_FIXTURES, RED_FIXTURES, BLUE_FIXTURES
    assert len(ALL_FIXTURES) == len(RED_FIXTURES) + len(BLUE_FIXTURES)


def test_all_fixtures_team_assignment():
    from atomics.eval.redblue.fixtures import RED_FIXTURES, BLUE_FIXTURES
    for f in RED_FIXTURES:
        assert f.team == "red", f"{f.id} should be red"
    for f in BLUE_FIXTURES:
        assert f.team == "blue", f"{f.id} should be blue"


def test_all_fixtures_ids_unique():
    from atomics.eval.redblue.fixtures import ALL_FIXTURES
    ids = [f.id for f in ALL_FIXTURES]
    assert len(ids) == len(set(ids)), "Duplicate Red/Blue fixture IDs"


def test_all_fixtures_valid_complexity():
    from atomics.eval.redblue.fixtures import ALL_FIXTURES
    valid = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    for f in ALL_FIXTURES:
        assert f.complexity in valid, f"{f.id}: invalid complexity {f.complexity}"


import asyncio
from unittest.mock import AsyncMock
from types import SimpleNamespace


def _provider(text="The Metasploit module search command is: search type:exploit"):
    p = AsyncMock()
    p.name = "mock"
    p.generate = AsyncMock(return_value=SimpleNamespace(
        text=text, model="mock-model", input_tokens=50, output_tokens=100,
        total_tokens=150, thinking_tokens=0, latency_ms=200.0,
        estimated_cost_usd=0.0, tokens_per_second=80.0,
    ))
    return p


def _judge(score=0.8, rationale="Correct and well-structured."):
    raw_score = int(score * 10)
    j = AsyncMock()
    j.name = "judge"
    j.generate = AsyncMock(return_value=SimpleNamespace(
        text=f"ACCURACY: {raw_score}\nCOMPLETENESS: {raw_score}\nFORMAT: {raw_score}\nRATIONALE: {rationale}",
        model="judge-model", input_tokens=20, output_tokens=30,
        total_tokens=50, thinking_tokens=0, latency_ms=80.0,
        estimated_cost_usd=0.0, tokens_per_second=100.0,
    ))
    return j


def test_run_redblue_returns_summary():
    from atomics.eval.redblue.runner import run_redblue, RedBlueSummary
    summary = asyncio.run(run_redblue(_provider(), judge_provider=_judge()))
    assert isinstance(summary, RedBlueSummary)
    assert summary.total_fixtures > 0


def test_run_redblue_red_only():
    from atomics.eval.redblue.runner import run_redblue
    from atomics.eval.redblue.fixtures import RED_FIXTURES
    summary = asyncio.run(run_redblue(_provider(), judge_provider=_judge(), mode="red"))
    assert summary.total_fixtures == len(RED_FIXTURES)


def test_run_redblue_blue_only():
    from atomics.eval.redblue.runner import run_redblue
    from atomics.eval.redblue.fixtures import BLUE_FIXTURES
    summary = asyncio.run(run_redblue(_provider(), judge_provider=_judge(), mode="blue"))
    assert summary.total_fixtures == len(BLUE_FIXTURES)


def test_run_redblue_per_category_scores():
    from atomics.eval.redblue.runner import run_redblue
    summary = asyncio.run(run_redblue(_provider(), judge_provider=_judge()))
    assert len(summary.category_scores) > 0


def test_cli_redblue_help():
    from click.testing import CliRunner
    from atomics.cli import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["redblue", "--help"])
    assert result.exit_code == 0
    assert "red" in result.output.lower()


# ── Multi-run (--runs N) support ─────────────────────────────────────────────

def test_run_redblue_runs_default_is_one():
    from atomics.eval.redblue.runner import run_redblue
    summary = asyncio.run(run_redblue(_provider(), judge_provider=_judge(), mode="red"))
    assert summary.runs == 1
    assert summary.quality_stddev is None


def test_run_redblue_multi_run_records_scores_and_stddev():
    """runs=3 records per-run scores; identical scores → stddev 0.0."""
    from atomics.eval.redblue.runner import run_redblue
    summary = asyncio.run(run_redblue(
        _provider(), judge_provider=_judge(score=0.8), mode="red", runs=3,
    ))
    assert summary.runs == 3
    assert summary.quality_stddev == 0.0
    for r in summary.results:
        if r.judge and not r.judge.parse_failed:
            assert len(r.run_scores) == 3


def test_run_redblue_multi_run_mean_written():
    """The mean across runs is written to the persisted accuracy score."""
    from atomics.eval.redblue.runner import run_redblue
    summary = asyncio.run(run_redblue(
        _provider(), judge_provider=_judge(score=0.6), mode="red", runs=2,
    ))
    for r in summary.results:
        if r.judge and not r.judge.parse_failed:
            assert abs(r.task_result.accuracy_score - r.judge.score) < 1e-6


def test_cli_redblue_runs_flag_present():
    from click.testing import CliRunner
    from atomics.cli import cli
    result = CliRunner().invoke(cli, ["redblue", "--help"])
    assert "--runs" in result.output


# ── --json-out ───────────────────────────────────────────────────────────────

def test_redblue_summary_to_dict_serializable():
    import json
    from atomics.eval.redblue.runner import run_redblue
    summary = asyncio.run(run_redblue(_provider(), judge_provider=_judge(), mode="red"))
    d = summary.to_dict()
    text = json.dumps(d)  # must round-trip
    assert '"overall_quality"' in text
    assert d["total_fixtures"] == len(d["fixtures"])
    assert d["mode"] == "red"
    f0 = d["fixtures"][0]
    for key in ("id", "team", "category", "status", "score", "rationale"):
        assert key in f0


def test_cli_redblue_json_out_flag_present():
    from click.testing import CliRunner
    from atomics.cli import cli
    result = CliRunner().invoke(cli, ["redblue", "--help"])
    assert "--json-out" in result.output


# ── Thinking-aware output budget ─────────────────────────────────────────────

def test_output_budget_expands_for_thinking_models():
    from atomics.eval.redblue.runner import _output_budget, _THINKING_MIN_OUTPUT_TOKENS
    from atomics.eval.redblue.fixtures import RED_FIXTURES
    fx = RED_FIXTURES[0]
    # explicit thinking=True → expanded
    assert _output_budget(fx, thinking=True, model="qwen2.5:7b") == max(
        fx.max_output_tokens, _THINKING_MIN_OUTPUT_TOKENS
    )
    # auto-detect a thinking-capable model → expanded
    assert _output_budget(fx, thinking=None, model="qwen3:14b") >= _THINKING_MIN_OUTPUT_TOKENS


def test_output_budget_unchanged_for_nonthinking():
    from atomics.eval.redblue.runner import _output_budget
    from atomics.eval.redblue.fixtures import RED_FIXTURES
    fx = RED_FIXTURES[0]
    assert _output_budget(fx, thinking=False, model="qwen2.5:7b") == fx.max_output_tokens
    assert _output_budget(fx, thinking=None, model="qwen2.5:7b") == fx.max_output_tokens


def test_run_redblue_all_runs_failed_records_failure():
    """When every run raises, the fixture is recorded as FAILED with no judge."""
    from atomics.eval.redblue.runner import run_redblue
    from atomics.models import TaskStatus

    p = AsyncMock()
    p.name = "mock"
    p.generate = AsyncMock(side_effect=RuntimeError("boom"))

    summary = asyncio.run(run_redblue(p, judge_provider=_judge(), mode="red", runs=2))
    assert summary.total_fixtures == len(summary.results)
    for r in summary.results:
        assert r.judge is None
        assert r.task_result.status == TaskStatus.FAILED
        assert r.task_result.error_class == "RuntimeError"
    # overall_quality is None when nothing scored
    assert summary.overall_quality is None


# ── RedBlueSummary computed properties ───────────────────────────────────────


import asyncio as _asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock


def _make_summary_with_results():
    """Build a RedBlueSummary with stubbed results for property testing."""
    from atomics.eval.redblue.runner import RedBlueSummary, RedBlueFixtureResult
    from atomics.eval.redblue.fixtures import RedBlueFixture
    from atomics.eval.judge import JudgeResult
    from atomics.models import TaskResult, TaskStatus, TaskCategory
    from datetime import UTC, datetime

    def _fixture(fid: str, team: str, category: str) -> RedBlueFixture:
        return RedBlueFixture(
            id=fid, team=team, category=category, complexity="MEDIUM",
            prompt="probe", gold_criteria=["a"],
        )

    def _task(latency: float = 200.0, cost: float = 0.001) -> TaskResult:
        t = TaskResult(
            run_id="test", category=TaskCategory.GENERAL_QA,
            task_name="x", provider="mock", model="m",
        )
        t.status = TaskStatus.SUCCESS
        t.latency_ms = latency
        t.estimated_cost_usd = cost
        return t

    def _judge(score: float = 0.8) -> JudgeResult:
        # score normalised: accuracy/4 * 0.4 + completeness/3 * 0.3 + format/3 * 0.3
        j = JudgeResult(
            score=score, accuracy=3, completeness=2, format_score=2,
            rationale="good", judge_model="m", parse_failed=False,
        )
        return j

    now = datetime.now(UTC)
    summary = RedBlueSummary(
        run_id="test", provider="mock", model="m", mode="all",
        started_at=now, completed_at=now,
    )
    summary.results = [
        RedBlueFixtureResult(fixture=_fixture("r1", "red", "recon"), task_result=_task(100.0, 0.001), judge=_judge(0.9)),
        RedBlueFixtureResult(fixture=_fixture("r2", "red", "recon"), task_result=_task(200.0, 0.002), judge=_judge(0.7)),
        RedBlueFixtureResult(fixture=_fixture("b1", "blue", "defense"), task_result=_task(300.0, 0.003), judge=_judge(0.8)),
        RedBlueFixtureResult(fixture=_fixture("f1", "red", "recon"), task_result=_task(150.0, 0.001), judge=None),
    ]
    return summary


def test_redblue_summary_overall_quality():
    summary = _make_summary_with_results()
    q = summary.overall_quality
    assert q is not None
    assert 0.0 <= q <= 1.0


def test_redblue_summary_overall_quality_no_results():
    from atomics.eval.redblue.runner import RedBlueSummary
    from datetime import UTC, datetime
    now = datetime.now(UTC)
    empty = RedBlueSummary(run_id="x", provider="m", model="m", mode="all",
                           started_at=now, completed_at=now, results=[])
    assert empty.overall_quality is None


def test_redblue_summary_category_scores():
    summary = _make_summary_with_results()
    cats = summary.category_scores
    assert "recon" in cats
    assert "defense" in cats
    assert 0.0 <= cats["recon"] <= 1.0


def test_redblue_summary_avg_latency_ms():
    summary = _make_summary_with_results()
    avg = summary.avg_latency_ms
    assert avg == round((100.0 + 200.0 + 300.0 + 150.0) / 4, 1)


def test_redblue_summary_avg_latency_empty():
    from atomics.eval.redblue.runner import RedBlueSummary
    from datetime import UTC, datetime
    now = datetime.now(UTC)
    empty = RedBlueSummary(run_id="x", provider="m", model="m", mode="all",
                           started_at=now, completed_at=now, results=[])
    assert empty.avg_latency_ms == 0.0


def test_redblue_summary_total_cost_usd():
    summary = _make_summary_with_results()
    assert abs(summary.total_cost_usd - 0.007) < 1e-9


# ── on_fixture_done callback paths ───────────────────────────────────────────


def _provider_raises():
    p = AsyncMock()
    p.name = "fail-prov"
    p.generate = AsyncMock(side_effect=RuntimeError("forced failure"))
    return p


def _provider_ok():
    p = AsyncMock()
    p.name = "ok-prov"
    p.generate = AsyncMock(return_value=SimpleNamespace(
        text="answer", model="m", input_tokens=20, output_tokens=40,
        total_tokens=60, thinking_tokens=0, latency_ms=150.0,
        estimated_cost_usd=0.0, tokens_per_second=100.0,
    ))
    return p


def _judge_good():
    j = AsyncMock()
    j.name = "judge"
    j.generate = AsyncMock(return_value=SimpleNamespace(
        text="ACCURACY: 8\nCOMPLETENESS: 7\nFORMAT: 8\nRATIONALE: good",
        model="judge-m", input_tokens=10, output_tokens=20,
        total_tokens=30, thinking_tokens=0, latency_ms=50.0,
        estimated_cost_usd=0.0, tokens_per_second=200.0,
    ))
    return j


def test_on_fixture_done_called_on_success_sync():
    from atomics.eval.redblue.runner import run_redblue

    received = []

    def cb(fr):
        received.append(fr)

    summary = _asyncio.run(run_redblue(
        _provider_ok(),
        judge_provider=_judge_good(),
        mode="red",
        on_fixture_done=cb,
    ))

    assert len(received) == summary.total_fixtures


def test_on_fixture_done_called_on_generate_failure_sync():
    from atomics.eval.redblue.runner import run_redblue
    from atomics.models import TaskStatus

    received = []

    def cb(fr):
        received.append(fr)

    summary = _asyncio.run(run_redblue(
        _provider_raises(),
        judge_provider=_judge_good(),
        mode="red",
        on_fixture_done=cb,
    ))

    assert len(received) == summary.total_fixtures
    for fr in received:
        assert fr.task_result.status == TaskStatus.FAILED


def test_on_fixture_done_called_async_callback():
    from atomics.eval.redblue.runner import run_redblue

    received = []

    async def async_cb(fr):
        received.append(fr)

    summary = _asyncio.run(run_redblue(
        _provider_ok(),
        judge_provider=_judge_good(),
        mode="red",
        on_fixture_done=async_cb,
    ))

    assert len(received) == summary.total_fixtures


def test_on_fixture_done_async_callback_on_failure():
    from atomics.eval.redblue.runner import run_redblue

    received = []

    async def async_cb(fr):
        received.append(fr)

    _asyncio.run(run_redblue(
        _provider_raises(),
        judge_provider=_judge_good(),
        mode="red",
        on_fixture_done=async_cb,
    ))

    assert len(received) > 0
