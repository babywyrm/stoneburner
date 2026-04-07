"""Tests for the SQLite metrics storage layer."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

from atomics.models import TaskCategory, TaskResult, TaskStatus
from atomics.storage.repository import MetricsRepository


def _tmp_repo() -> MetricsRepository:
    tmp = tempfile.mktemp(suffix=".db")
    return MetricsRepository(Path(tmp))


def test_create_and_complete_run():
    repo = _tmp_repo()
    repo.create_run("run-001")
    summary = repo.complete_run("run-001")
    assert summary.run_id == "run-001"
    assert summary.total_tasks == 0
    repo.close()


def test_save_and_query_task_result():
    repo = _tmp_repo()
    repo.create_run("run-002")

    result = TaskResult(
        run_id="run-002",
        category=TaskCategory.GENERAL_QA,
        task_name="quick_question",
        provider="claude",
        model="claude-sonnet-4-20250514",
        status=TaskStatus.SUCCESS,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        latency_ms=500.0,
        estimated_cost_usd=0.001,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    repo.save_task_result(result)

    tasks = repo.get_run_tasks("run-002")
    assert len(tasks) == 1
    assert tasks[0]["task_name"] == "quick_question"
    assert tasks[0]["total_tokens"] == 150

    summary = repo.complete_run("run-002")
    assert summary.total_tasks == 1
    assert summary.successful_tasks == 1
    assert summary.total_tokens == 150
    repo.close()


def test_usage_by_category():
    repo = _tmp_repo()
    repo.create_run("run-003")

    categories = [TaskCategory.RESEARCH, TaskCategory.RESEARCH, TaskCategory.SECURITY_NEWS]
    for i, cat in enumerate(categories):
        result = TaskResult(
            task_id=f"t{i}",
            run_id="run-003",
            category=cat,
            task_name=f"task_{i}",
            provider="claude",
            model="test",
            status=TaskStatus.SUCCESS,
            total_tokens=100 * (i + 1),
            estimated_cost_usd=0.001 * (i + 1),
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
        repo.save_task_result(result)

    breakdown = repo.get_usage_by_category()
    assert len(breakdown) == 2
    cats = {r["category"] for r in breakdown}
    assert "research" in cats
    assert "security_news" in cats
    repo.close()


def test_recent_runs():
    repo = _tmp_repo()
    for i in range(5):
        repo.create_run(f"run-{i:03d}")
        repo.complete_run(f"run-{i:03d}")

    runs = repo.get_recent_runs(limit=3)
    assert len(runs) == 3
    repo.close()


def test_query_task_results_since_hours():
    repo = _tmp_repo()
    repo.create_run("exp-1")
    result = TaskResult(
        run_id="exp-1",
        category=TaskCategory.GENERAL_QA,
        task_name="t1",
        provider="claude",
        model="m",
        status=TaskStatus.SUCCESS,
        total_tokens=10,
        estimated_cost_usd=0.0,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    repo.save_task_result(result)
    rows = repo.query_task_results(since_hours=24.0, limit=10)
    assert len(rows) == 1
    assert rows[0]["task_name"] == "t1"
    repo.close()


def test_get_hourly_token_rate_sums_recent_tasks():
    repo = _tmp_repo()
    repo.create_run("hourly-run")
    result = TaskResult(
        run_id="hourly-run",
        category=TaskCategory.GENERAL_QA,
        task_name="t1",
        provider="claude",
        model="test",
        status=TaskStatus.SUCCESS,
        total_tokens=42,
        estimated_cost_usd=0.0,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    repo.save_task_result(result)
    repo.complete_run("hourly-run")

    rate = repo.get_hourly_token_rate()
    assert rate == 42.0
    repo.close()
