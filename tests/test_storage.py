"""Tests for the SQLite metrics storage layer."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

from atomics.models import TaskCategory, TaskResult, TaskStatus
from atomics.storage.repository import MetricsRepository, _percentile
from atomics.storage.schema import SCHEMA_VERSION, init_db


def _tmp_repo() -> MetricsRepository:
    tmp = tempfile.mktemp(suffix=".db")
    return MetricsRepository(Path(tmp))


def test_schema_version_is_2():
    assert SCHEMA_VERSION == 2


def test_schema_fresh_start_on_version_mismatch(tmp_path):
    """A v1 database gets wiped and recreated as v2."""
    import sqlite3

    db_path = tmp_path / "migrate.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
        INSERT INTO schema_version (version) VALUES (1);
        CREATE TABLE runs (run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL);
        INSERT INTO runs (run_id, started_at) VALUES ('old', '2026-01-01');
    """)
    conn.commit()
    conn.close()

    conn2 = init_db(db_path)
    rows = conn2.execute("SELECT * FROM runs").fetchall()
    assert len(rows) == 0
    ver = conn2.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    assert ver == SCHEMA_VERSION
    cols = [desc[0] for desc in conn2.execute("SELECT * FROM runs").description]
    assert "tier" in cols
    assert "provider" in cols
    assert "trigger" in cols
    conn2.close()


def test_schedules_table_exists(tmp_path):
    db_path = tmp_path / "sched.db"
    conn = init_db(db_path)
    conn.execute("SELECT * FROM schedules")
    conn.close()


def test_create_run_with_metadata():
    repo = _tmp_repo()
    repo.create_run("run-meta", tier="mega", provider="openai", model="gpt-4o", trigger="scheduled")
    runs = repo.get_recent_runs(limit=1)
    assert runs[0]["tier"] == "mega"
    assert runs[0]["provider"] == "openai"
    assert runs[0]["model"] == "gpt-4o"
    assert runs[0]["trigger"] == "scheduled"
    repo.close()


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


def test_compare_providers():
    repo = _tmp_repo()
    for provider, run_id in [("claude", "cmp-1"), ("bedrock", "cmp-2")]:
        repo.create_run(run_id, provider=provider)
        for i in range(3):
            result = TaskResult(
                task_id=f"{run_id}-t{i}",
                run_id=run_id,
                category=TaskCategory.GENERAL_QA,
                task_name="q",
                provider=provider,
                model="m",
                status=TaskStatus.SUCCESS,
                total_tokens=100,
                latency_ms=200.0 + i * 100,
                estimated_cost_usd=0.01,
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
            )
            repo.save_task_result(result)
        repo.complete_run(run_id)

    rows = repo.compare_providers()
    assert len(rows) == 2
    providers = {r["group_key"] for r in rows}
    assert "claude" in providers
    assert "bedrock" in providers
    for r in rows:
        assert r["task_count"] == 3
        assert r["successes"] == 3
        assert "p50_latency_ms" in r
        assert "p95_latency_ms" in r
        assert "cost_per_1k_tokens" in r
        assert r["p50_latency_ms"] > 0
        assert r["cost_per_1k_tokens"] > 0
    repo.close()


def test_compare_providers_by_model():
    repo = _tmp_repo()
    repo.create_run("mdl-1", provider="openai")
    for model, tid in [("gpt-4o", "m1"), ("gpt-4o-mini", "m2")]:
        result = TaskResult(
            task_id=tid,
            run_id="mdl-1",
            category=TaskCategory.GENERAL_QA,
            task_name="q",
            provider="openai",
            model=model,
            status=TaskStatus.SUCCESS,
            total_tokens=50,
            latency_ms=100.0,
            estimated_cost_usd=0.005,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
        repo.save_task_result(result)
    repo.complete_run("mdl-1")

    rows = repo.compare_providers(group_by="model")
    models = {r["group_key"] for r in rows}
    assert "gpt-4o" in models
    assert "gpt-4o-mini" in models
    repo.close()


def test_get_runs_by_provider():
    repo = _tmp_repo()
    repo.create_run("rbp-1", provider="claude")
    repo.complete_run("rbp-1")
    repo.create_run("rbp-2", provider="bedrock")
    repo.complete_run("rbp-2")

    rows = repo.get_runs_by_provider()
    assert len(rows) == 2
    providers = {r["provider"] for r in rows}
    assert "claude" in providers
    assert "bedrock" in providers
    repo.close()


def test_schedule_crud():
    repo = _tmp_repo()
    repo.save_schedule(
        schedule_id="launchd.ez.bedrock",
        format="launchd",
        tier="ez",
        provider="bedrock",
        model=None,
        interval_minutes=30,
        max_iterations=10,
    )

    schedules = repo.get_schedules()
    assert len(schedules) == 1
    assert schedules[0]["schedule_id"] == "launchd.ez.bedrock"
    assert schedules[0]["tier"] == "ez"
    assert schedules[0]["provider"] == "bedrock"
    assert schedules[0]["interval_minutes"] == 30

    repo.update_schedule_last_run("launchd.ez.bedrock", "success")
    schedules = repo.get_schedules()
    assert schedules[0]["last_status"] == "success"
    assert schedules[0]["last_run_at"] is not None

    repo.remove_schedule("launchd.ez.bedrock")
    assert len(repo.get_schedules()) == 0
    repo.close()


def test_percentile_basic():
    assert _percentile([], 50) == 0.0
    assert _percentile([10.0], 50) == 10.0
    assert _percentile([10.0, 20.0], 50) == 15.0
    assert _percentile([10.0, 20.0, 30.0, 40.0, 50.0], 50) == 30.0


def test_percentile_p95():
    vals = sorted([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    p95 = _percentile(vals, 95)
    assert 900 < p95 <= 1000


def test_compare_cost_per_1k_tokens():
    repo = _tmp_repo()
    repo.create_run("cost-run", provider="openai")
    result = TaskResult(
        task_id="cost-t1",
        run_id="cost-run",
        category=TaskCategory.GENERAL_QA,
        task_name="q",
        provider="openai",
        model="gpt-4o",
        status=TaskStatus.SUCCESS,
        total_tokens=1000,
        latency_ms=500.0,
        estimated_cost_usd=0.01,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    repo.save_task_result(result)
    repo.complete_run("cost-run")

    rows = repo.compare_providers()
    assert len(rows) == 1
    assert abs(rows[0]["cost_per_1k_tokens"] - 0.01) < 0.0001
    repo.close()


def test_compare_latency_percentiles():
    repo = _tmp_repo()
    repo.create_run("lat-run", provider="claude")
    latencies = [100, 150, 200, 250, 300, 350, 400, 450, 500, 1000]
    for i, lat in enumerate(latencies):
        result = TaskResult(
            task_id=f"lat-t{i}",
            run_id="lat-run",
            category=TaskCategory.GENERAL_QA,
            task_name="q",
            provider="claude",
            model="m",
            status=TaskStatus.SUCCESS,
            total_tokens=100,
            latency_ms=float(lat),
            estimated_cost_usd=0.001,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
        repo.save_task_result(result)
    repo.complete_run("lat-run")

    rows = repo.compare_providers()
    assert len(rows) == 1
    r = rows[0]
    assert 200 < r["p50_latency_ms"] < 400
    assert r["p95_latency_ms"] > r["p50_latency_ms"]
    assert r["p95_latency_ms"] >= 500
    repo.close()
