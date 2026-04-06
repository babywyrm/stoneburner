"""Tests for the reporting module."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from atomics.models import TaskCategory, TaskResult, TaskStatus
from atomics.reporting import print_category_breakdown, print_hourly_usage, print_recent_runs
from atomics.storage.repository import MetricsRepository


def _seeded_repo() -> MetricsRepository:
    repo = MetricsRepository(Path(tempfile.mktemp(suffix=".db")))
    repo.create_run("rpt-001")
    for i in range(3):
        result = TaskResult(
            task_id=f"rpt-t{i}",
            run_id="rpt-001",
            category=TaskCategory.RESEARCH if i < 2 else TaskCategory.SECURITY_NEWS,
            task_name=f"task_{i}",
            provider="claude",
            model="test",
            status=TaskStatus.SUCCESS,
            input_tokens=50,
            output_tokens=100 * (i + 1),
            total_tokens=50 + 100 * (i + 1),
            latency_ms=500.0 + i * 100,
            estimated_cost_usd=0.01 * (i + 1),
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )
        repo.save_task_result(result)
    repo.complete_run("rpt-001")
    return repo


def test_print_recent_runs(capsys):
    repo = _seeded_repo()
    print_recent_runs(repo, limit=5)
    captured = capsys.readouterr()
    assert "rpt-001" in captured.out
    assert "3" in captured.out
    repo.close()


def test_print_recent_runs_empty(capsys):
    repo = MetricsRepository(Path(tempfile.mktemp(suffix=".db")))
    print_recent_runs(repo)
    captured = capsys.readouterr()
    assert "No runs" in captured.out
    repo.close()


def test_print_hourly_usage(capsys):
    repo = _seeded_repo()
    print_hourly_usage(repo, hours=24)
    captured = capsys.readouterr()
    assert "Token Usage" in captured.out
    repo.close()


def test_print_hourly_usage_empty(capsys):
    repo = MetricsRepository(Path(tempfile.mktemp(suffix=".db")))
    print_hourly_usage(repo)
    captured = capsys.readouterr()
    assert "No hourly" in captured.out
    repo.close()


def test_print_category_breakdown(capsys):
    repo = _seeded_repo()
    print_category_breakdown(repo)
    captured = capsys.readouterr()
    assert "research" in captured.out
    assert "security_news" in captured.out
    repo.close()


def test_print_category_breakdown_empty(capsys):
    repo = MetricsRepository(Path(tempfile.mktemp(suffix=".db")))
    print_category_breakdown(repo)
    captured = capsys.readouterr()
    assert "No category" in captured.out
    repo.close()
