"""Tests for the cost optimization advisor."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

from atomics.advisor import analyze_cost_optimization
from atomics.models import TaskCategory, TaskResult, TaskStatus
from atomics.storage.repository import MetricsRepository
from atomics.storage.schema import init_db


def _seeded_db() -> tuple[Path, MetricsRepository]:
    """Create a DB with multi-model eval data for advisor testing."""
    db_path = Path(tempfile.mktemp(suffix=".db"))
    repo = MetricsRepository(db_path)

    models = [
        ("claude-sonnet-4-6", 0.95, 0.005),
        ("claude-haiku-4-5", 0.88, 0.001),
        ("gpt-4o", 0.92, 0.004),
        ("gpt-4o-mini", 0.82, 0.0008),
    ]

    for model, quality, cost in models:
        run_id = f"run-{model}"
        repo.create_run(run_id, provider="test", model=model)
        for i in range(5):
            result = TaskResult(
                task_id=f"{model}-t{i}",
                run_id=run_id,
                category=TaskCategory.GENERAL_QA,
                task_name=f"task_{i}",
                provider="test",
                model=model,
                status=TaskStatus.SUCCESS,
                total_tokens=100,
                latency_ms=500.0,
                estimated_cost_usd=cost,
                accuracy_score=quality,
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
            )
            repo.save_task_result(result)
        repo.complete_run(run_id)

    return db_path, repo


def test_advisor_finds_cheaper_model():
    db_path, repo = _seeded_db()
    conn = init_db(db_path)
    try:
        summary = analyze_cost_optimization(conn, min_quality=0.8)
        assert len(summary.recommendations) >= 1
        for r in summary.recommendations:
            assert r.recommended_cost_per_task < r.current_cost_per_task
            assert r.recommended_quality >= 0.8
            assert r.cost_savings_pct > 0
    finally:
        conn.close()
        repo.close()


def test_advisor_respects_quality_threshold():
    db_path, repo = _seeded_db()
    conn = init_db(db_path)
    try:
        summary = analyze_cost_optimization(conn, min_quality=0.95)
        for r in summary.recommendations:
            assert r.recommended_quality >= 0.95
    finally:
        conn.close()
        repo.close()


def test_advisor_no_data():
    db_path = Path(tempfile.mktemp(suffix=".db"))
    conn = init_db(db_path)
    try:
        summary = analyze_cost_optimization(conn, min_quality=0.8)
        assert len(summary.recommendations) == 0
        assert summary.models_analyzed == 0
    finally:
        conn.close()


def test_advisor_single_model_no_recommendations():
    db_path = Path(tempfile.mktemp(suffix=".db"))
    repo = MetricsRepository(db_path)
    repo.create_run("r1", provider="test", model="only-model")
    for i in range(3):
        result = TaskResult(
            task_id=f"t{i}",
            run_id="r1",
            category=TaskCategory.GENERAL_QA,
            task_name=f"task_{i}",
            provider="test",
            model="only-model",
            status=TaskStatus.SUCCESS,
            total_tokens=100,
            estimated_cost_usd=0.01,
            accuracy_score=0.9,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
        repo.save_task_result(result)
    repo.complete_run("r1")
    repo.close()

    conn = init_db(db_path)
    try:
        summary = analyze_cost_optimization(conn)
        assert len(summary.recommendations) == 0
    finally:
        conn.close()


def test_advisor_with_current_model():
    db_path, repo = _seeded_db()
    conn = init_db(db_path)
    try:
        summary = analyze_cost_optimization(
            conn, min_quality=0.8, current_model="claude-sonnet-4-6"
        )
        for r in summary.recommendations:
            assert r.current_model == "claude-sonnet-4-6"
    finally:
        conn.close()
        repo.close()


def test_advisor_summary_to_dict():
    db_path, repo = _seeded_db()
    conn = init_db(db_path)
    try:
        summary = analyze_cost_optimization(conn, min_quality=0.8)
        d = summary.to_dict()
        assert "total_current_cost" in d
        assert "total_recommended_cost" in d
        assert "overall_savings_pct" in d
        assert "recommendations" in d
        assert isinstance(d["recommendations"], list)
    finally:
        conn.close()
        repo.close()


def test_advisor_overall_savings():
    db_path, repo = _seeded_db()
    conn = init_db(db_path)
    try:
        summary = analyze_cost_optimization(conn, min_quality=0.8)
        if summary.recommendations:
            assert summary.total_recommended_cost <= summary.total_current_cost
            assert summary.overall_savings_pct >= 0
    finally:
        conn.close()
        repo.close()


def test_cli_advisor_help():
    from click.testing import CliRunner

    from atomics.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["advisor", "--help"])
    assert result.exit_code == 0
    assert "--min-quality" in result.output
    assert "--current-model" in result.output


def test_cli_advisor_empty_db(tmp_path):
    from click.testing import CliRunner

    from atomics.cli import cli

    runner = CliRunner(env={"ATOMICS_DB_PATH": str(tmp_path / "empty.db")})
    result = runner.invoke(cli, ["advisor"])
    assert result.exit_code == 0
    assert "No optimization" in result.output
