"""Tests for the SQLite metrics storage layer."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from atomics.models import TaskCategory, TaskResult, TaskStatus
from atomics.storage import EvaluationResultRecord
from atomics.storage.repository import MetricsRepository, _percentile
from atomics.storage.schema import SCHEMA_VERSION, init_db


def _tmp_repo() -> MetricsRepository:
    tmp = tempfile.mktemp(suffix=".db")
    return MetricsRepository(Path(tmp))


def test_schema_version_is_current():
    assert SCHEMA_VERSION == 20


def test_archreview_results_table_exists(tmp_path):
    conn = init_db(tmp_path / "t.db")
    conn.execute(
        "INSERT INTO archreview_results "
        "(result_id, run_id, repo, tier, model, provider, round, "
        " objective_recall, objective_precision, objective_f, judge_score, "
        " finding_count, parse_failed, tokens_in, tokens_out, cost_usd, "
        " latency_ms, findings_json, matched_categories_json, timestamp) "
        "VALUES ('r1','run1','juice-shop','floor','qwen2.5:14b','ollama',1, "
        " 0.7,0.8,0.74,0.66,11,0,1200,400,0.0,8300.0,'[]','[]','2026-01-01')"
    )
    row = conn.execute(
        "SELECT objective_recall FROM archreview_results WHERE result_id='r1'"
    ).fetchone()
    assert row[0] == 0.7


def test_save_archreview_result_roundtrip(tmp_path):
    from atomics.archreview.models import ArchReviewResult, Finding
    from atomics.storage.repository import MetricsRepository

    repo = MetricsRepository(tmp_path / "m.db")
    r = ArchReviewResult(
        run_id="run1", repo="juice-shop", tier="floor", model="qwen2.5:14b",
        provider="ollama", round=1,
        findings=[Finding("injection", "a.ts", "high", "raw sql")],
        objective_recall=0.7, objective_precision=0.8, objective_f=0.74,
        judge_score=0.66, matched_categories=["injection"], tokens_in=1200,
        tokens_out=400, latency_ms=8300.0, pack_hash="abc123",
    )
    repo.save_archreview_result(r)
    row = repo._conn.execute(
        "SELECT repo, objective_recall, finding_count, matched_categories_json "
        "FROM archreview_results WHERE run_id='run1'"
    ).fetchone()
    assert row[0] == "juice-shop"
    assert row[1] == 0.7
    assert row[2] == 1
    assert "injection" in row[3]


def test_adversarial_results_table_exists(tmp_path):
    conn = init_db(tmp_path / "test.db")
    conn.execute(
        "INSERT INTO runs (run_id, started_at) VALUES ('run1', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO adversarial_results "
        "(result_id, run_id, fixture_id, category, severity, provider, model, timestamp) "
        "VALUES ('r1','run1','f1','prompt_injection','HIGH','ollama','qwen3:14b','2026-01-01')"
    )
    row = conn.execute("SELECT fixture_id FROM adversarial_results WHERE result_id='r1'").fetchone()
    assert row[0] == "f1"
    conn.close()


def test_probe_results_table_exists(tmp_path):
    conn = init_db(tmp_path / "test.db")
    conn.execute(
        "INSERT INTO probe_results "
        "(result_id, run_id, target_name, artifact_type, check_id, provider, model, timestamp) "
        "VALUES ('p1','run1','my-target','access-log','ioc_analysis','ollama',"
        "'qwen3:14b','2026-01-01')"
    )
    row = conn.execute("SELECT target_name FROM probe_results WHERE result_id='p1'").fetchone()
    assert row[0] == "my-target"


def test_task_results_has_suite_column(tmp_path):
    conn = init_db(tmp_path / "test.db")
    conn.execute(
        "INSERT INTO runs (run_id, started_at, provider, model, tier, trigger) "
        "VALUES ('run1','2026-01-01','ollama','qwen3:14b','baseline','manual')"
    )
    conn.execute(
        "INSERT INTO task_results "
        "(task_id, run_id, category, task_name, provider, model, status, started_at, suite) "
        "VALUES ('t1','run1','general_qa','ev-01','ollama','qwen3:14b','success',"
        "'2026-01-01','redblue-red')"
    )
    conn.commit()
    row = conn.execute("SELECT suite FROM task_results WHERE task_id='t1'").fetchone()
    assert row[0] == "redblue-red"


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


def test_schema_migration_backs_up_before_wipe(tmp_path):
    """The v19 pre-wipe backup exists and still holds the old data."""
    import sqlite3

    db_path = tmp_path / "migrate.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
        INSERT INTO schema_version (version) VALUES (19);
        CREATE TABLE runs (run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL);
        INSERT INTO runs (run_id, started_at) VALUES ('keepme', '2026-01-01');
    """)
    conn.commit()
    conn.close()

    conn2 = init_db(db_path)
    conn2.close()

    backups = list(tmp_path.glob("migrate.db.v19.*.bak"))
    assert len(backups) == 1, f"expected one backup, found {backups}"

    # The backup still contains the pre-migration row.
    bconn = sqlite3.connect(str(backups[0]))
    old = bconn.execute("SELECT run_id FROM runs").fetchall()
    bconn.close()
    assert old == [("keepme",)]


def test_schema_migration_aborts_and_closes_when_backup_fails(
    tmp_path, monkeypatch
):
    import sqlite3

    from atomics.storage import schema

    db_path = tmp_path / "backup-failure.db"
    setup = sqlite3.connect(db_path)
    setup.executescript("""
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
        INSERT INTO schema_version (version) VALUES (17);
        CREATE TABLE runs (run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL);
        INSERT INTO runs (run_id, started_at) VALUES ('sentinel', '2026-01-01');
    """)
    setup.close()

    opened = []
    real_connect = schema.sqlite3.connect

    def capture_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(schema.sqlite3, "connect", capture_connect)
    monkeypatch.setattr(
        schema,
        "_backup_before_wipe",
        lambda *_args: (_ for _ in ()).throw(sqlite3.OperationalError("disk full")),
    )

    with pytest.raises(sqlite3.OperationalError, match="disk full"):
        init_db(db_path)

    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute("SELECT 1")
    verify = real_connect(db_path)
    try:
        assert verify.execute(
            "SELECT version FROM schema_version"
        ).fetchone()[0] == 17
        assert verify.execute("SELECT run_id FROM runs").fetchone()[0] == "sentinel"
        verify.execute(
            "INSERT INTO runs (run_id, started_at) "
            "VALUES ('after-failure', '2026-01-02')"
        )
        verify.commit()
    finally:
        verify.close()


def test_schema_migration_rolls_back_mid_reset_failure(tmp_path, monkeypatch):
    import sqlite3

    from atomics.storage import schema

    db_path = tmp_path / "reset-failure.db"
    setup = sqlite3.connect(db_path)
    setup.executescript("""
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
        INSERT INTO schema_version (version) VALUES (17);
        CREATE TABLE runs (run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL);
        INSERT INTO runs (run_id, started_at) VALUES ('sentinel', '2026-01-01');
    """)
    setup.close()
    monkeypatch.setattr(
        schema,
        "SCHEMA_SQL",
        "CREATE TABLE replacement (id INTEGER); INVALID SCHEMA STATEMENT;",
    )

    with pytest.raises(sqlite3.OperationalError):
        init_db(db_path)

    verify = sqlite3.connect(db_path)
    try:
        assert verify.execute(
            "SELECT version FROM schema_version"
        ).fetchone()[0] == 17
        assert verify.execute("SELECT run_id FROM runs").fetchone()[0] == "sentinel"
        assert verify.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name='replacement'"
        ).fetchone()[0] == 0
    finally:
        verify.close()


def test_schema_migration_lock_prevents_post_backup_write_loss(
    tmp_path, monkeypatch
):
    import sqlite3
    import threading

    from atomics.storage import schema

    db_path = tmp_path / "migration-race.db"
    setup = sqlite3.connect(db_path)
    setup.executescript("""
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
        INSERT INTO schema_version (version) VALUES (17);
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            legacy_only TEXT
        );
        INSERT INTO runs (run_id, started_at, legacy_only)
        VALUES ('sentinel', '2026-01-01', 'before-backup');
    """)
    setup.close()

    backup_finished = threading.Event()
    writer_attempting = threading.Event()
    writer_finished = threading.Event()
    writer_result = {}
    real_backup = schema._backup_before_wipe

    def synchronized_backup(conn, path, current):
        backup_path = real_backup(conn, path, current)
        backup_finished.set()
        assert writer_attempting.wait(timeout=2)
        writer_finished.wait(timeout=0.25)
        return backup_path

    def concurrent_writer():
        conn = sqlite3.connect(db_path, timeout=5)
        try:
            assert backup_finished.wait(timeout=2)
            writer_attempting.set()
            conn.execute(
                "INSERT INTO runs (run_id, started_at, legacy_only) "
                "VALUES ('racing-writer', '2026-01-02', 'after-backup')"
            )
            conn.commit()
            writer_result["committed"] = True
        except sqlite3.Error as exc:
            writer_result["error"] = exc
        finally:
            conn.close()
            writer_finished.set()

    monkeypatch.setattr(schema, "_backup_before_wipe", synchronized_backup)
    writer = threading.Thread(target=concurrent_writer)
    writer.start()
    migrated = init_db(db_path)
    migrated.close()
    writer.join(timeout=6)
    assert not writer.is_alive()

    backup_path = next(tmp_path.glob("migration-race.db.v17.*.bak"))
    backup = sqlite3.connect(backup_path)
    live = sqlite3.connect(db_path)
    try:
        in_backup = backup.execute(
            "SELECT COUNT(*) FROM runs WHERE run_id='racing-writer'"
        ).fetchone()[0]
        in_live = live.execute(
            "SELECT COUNT(*) FROM runs WHERE run_id='racing-writer'"
        ).fetchone()[0]
    finally:
        backup.close()
        live.close()

    assert not writer_result.get("committed") or in_backup or in_live
    assert writer_result.get("committed") is not True
    assert isinstance(writer_result.get("error"), sqlite3.Error)


def test_schema_no_backup_on_fresh_db(tmp_path):
    """A brand-new DB (version 0) is not backed up — nothing to preserve."""
    db_path = tmp_path / "fresh.db"
    conn = init_db(db_path)
    conn.close()
    assert list(tmp_path.glob("*.bak")) == []


def test_complete_probe_run_finalizes_parent(tmp_path):
    """probe runs get a parent row that completion finalizes with a count."""
    repo = MetricsRepository(tmp_path / "probe.db")
    repo.create_run("p1", tier="probe", provider="ollama", model="m")

    class _R:
        target_name = "t"
        artifact_type = "log"
        check_id = "c"
        score = 0.9
        prev_score = None
        regressed = False
        judge_model = "j"
        judge_rationale = "ok"
        thinking_enabled = False
        thinking_tokens = 0

    repo.save_probe_result("p1", _R())
    repo.complete_probe_run("p1")

    run = repo._conn.execute(
        "SELECT completed_at, total_tasks, tier FROM runs WHERE run_id='p1'"
    ).fetchone()
    assert run["completed_at"] is not None
    assert run["total_tasks"] == 1
    assert run["tier"] == "probe"


def test_complete_archreview_run_finalizes_parent(tmp_path):
    """archreview runs get a parent row that completion finalizes."""
    from atomics.archreview.models import ArchReviewResult

    repo = MetricsRepository(tmp_path / "arch.db")
    repo.create_run("a1", tier="archreview", provider="ollama", model="m")
    repo.save_archreview_result(ArchReviewResult(
        run_id="a1", repo="juice-shop", tier="floor", model="m",
        provider="ollama", round=0, findings=[],
    ))
    repo.complete_archreview_run("a1")

    run = repo._conn.execute(
        "SELECT completed_at, total_tasks, tier FROM runs WHERE run_id='a1'"
    ).fetchone()
    assert run["completed_at"] is not None
    assert run["total_tasks"] == 1
    assert run["tier"] == "archreview"


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


def test_query_task_results_suite_filter():
    """query_task_results isolates a suite (exact) and a prefix (redblue-*)."""
    repo = _tmp_repo()
    repo.create_run("run-suite")

    def _mk(name, suite):
        r = TaskResult(
            run_id="run-suite", category=TaskCategory.GENERAL_QA, task_name=name,
            provider="ollama", model="m", status=TaskStatus.SUCCESS,
            started_at=datetime.now(UTC), completed_at=datetime.now(UTC),
        )
        repo.save_task_result(r, suite=suite)

    _mk("ev-1", "eval")
    _mk("rb-red-1", "redblue-red")
    _mk("rb-blue-1", "redblue-blue")

    all_rows = repo.query_task_results()
    assert len(all_rows) == 3
    eval_rows = repo.query_task_results(suite="eval")
    assert len(eval_rows) == 1 and eval_rows[0]["task_name"] == "ev-1"
    redblue_rows = repo.query_task_results(suite_prefix="redblue-")
    assert len(redblue_rows) == 2
    assert {r["suite"] for r in redblue_rows} == {"redblue-red", "redblue-blue"}


def test_cli_export_suite_choices_include_eval_redblue():
    from click.testing import CliRunner

    from atomics.cli import cli
    result = CliRunner().invoke(cli, ["export", "--help"])
    assert result.exit_code == 0
    assert "redblue" in result.output
    assert "eval" in result.output


def test_save_and_query_task_result():
    repo = _tmp_repo()
    repo.create_run("run-002")

    result = TaskResult(
        run_id="run-002",
        category=TaskCategory.GENERAL_QA,
        task_name="quick_question",
        provider="claude",
        model="claude-sonnet-4-6",
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


def test_save_and_query_cache_and_tps_basis():
    """Cache tokens and tps_basis must round-trip through the DB."""
    repo = _tmp_repo()
    repo.create_run("run-cache")

    result = TaskResult(
        run_id="run-cache",
        category=TaskCategory.GENERAL_QA,
        task_name="cached_q",
        provider="claude",
        model="claude-sonnet-4-6",
        status=TaskStatus.SUCCESS,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        cache_read_tokens=2000,
        cache_write_tokens=400,
        tokens_per_second=42.0,
        tps_basis="generation",
        estimated_cost_usd=0.001,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    repo.save_task_result(result)

    tasks = repo.get_run_tasks("run-cache")
    assert tasks[0]["cache_read_tokens"] == 2000
    assert tasks[0]["cache_write_tokens"] == 400
    assert tasks[0]["tps_basis"] == "generation"

    rows = repo.compare_providers()
    claude_row = next(r for r in rows if r["group_key"] == "claude")
    assert claude_row["total_cache_read_tokens"] == 2000
    assert claude_row["total_cache_write_tokens"] == 400
    assert claude_row["tps_bases"] == "generation"
    assert claude_row["avg_thinking_tokens"] == 0
    repo.close()


def test_save_and_query_criteria_coverage():
    """criteria_coverage must round-trip and aggregate as an average."""
    repo = _tmp_repo()
    repo.create_run("run-cov")

    result = TaskResult(
        run_id="run-cov",
        category=TaskCategory.GENERAL_QA,
        task_name="covered_q",
        provider="ollama",
        model="qwen2.5:7b",
        status=TaskStatus.SUCCESS,
        accuracy_score=0.8,
        criteria_coverage=0.667,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    repo.save_task_result(result)

    tasks = repo.get_run_tasks("run-cov")
    assert tasks[0]["criteria_coverage"] == 0.667

    rows = repo.compare_providers()
    row = next(r for r in rows if r["group_key"] == "ollama")
    assert row["avg_criteria_coverage"] == pytest.approx(0.667)
    repo.close()


def test_save_and_query_judge_score_stdev():
    """Consensus inter-judge stdev must round-trip and aggregate."""
    repo = _tmp_repo()
    repo.create_run("run-stdev")
    repo.save_task_result(
        TaskResult(
            run_id="run-stdev",
            category=TaskCategory.GENERAL_QA,
            task_name="q",
            provider="claude",
            model="claude-sonnet-4-6",
            status=TaskStatus.SUCCESS,
            accuracy_score=0.7,
            judge_score_stdev=0.3,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
    )
    tasks = repo.get_run_tasks("run-stdev")
    assert tasks[0]["judge_score_stdev"] == 0.3

    rows = repo.compare_providers()
    row = next(r for r in rows if r["group_key"] == "claude")
    assert row["avg_judge_score_stdev"] == pytest.approx(0.3)
    repo.close()


def test_criteria_coverage_defaults_to_null():
    """A task with no gold criteria persists NULL coverage."""
    repo = _tmp_repo()
    repo.create_run("run-nocov")
    repo.save_task_result(
        TaskResult(
            run_id="run-nocov",
            category=TaskCategory.GENERAL_QA,
            task_name="q",
            provider="ollama",
            model="qwen2.5:7b",
            status=TaskStatus.SUCCESS,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
    )
    tasks = repo.get_run_tasks("run-nocov")
    assert tasks[0]["criteria_coverage"] is None
    repo.close()


def test_cache_and_tps_basis_defaults():
    """Defaults persist as 0 / wall_clock when a provider reports no cache use."""
    repo = _tmp_repo()
    repo.create_run("run-nocache")
    repo.save_task_result(
        TaskResult(
            run_id="run-nocache",
            category=TaskCategory.GENERAL_QA,
            task_name="plain",
            provider="ollama",
            model="qwen2.5:7b",
            status=TaskStatus.SUCCESS,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
    )
    tasks = repo.get_run_tasks("run-nocache")
    assert tasks[0]["cache_read_tokens"] == 0
    assert tasks[0]["cache_write_tokens"] == 0
    assert tasks[0]["tps_basis"] == "wall_clock"
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


# ── Stress result persistence ────────────────────────────────────────────────

def test_stress_results_table_exists(tmp_path):
    conn = init_db(tmp_path / "test.db")
    conn.execute(
        "INSERT INTO stress_results "
        "(result_id, model, host, peak_tps, saturation_concurrency, "
        "duration_seconds, total_tokens, total_requests, timestamp) "
        "VALUES ('s1','qwen2.5:7b','http://localhost:11434',45.2,4,"
        "60.0,5000,20,'2026-01-01')"
    )
    row = conn.execute("SELECT model FROM stress_results WHERE result_id='s1'").fetchone()
    assert row[0] == "qwen2.5:7b"


def test_save_stress_result():
    from atomics.stress import ConcurrencyResult, StressResult

    repo = _tmp_repo()

    sr = StressResult(
        model="qwen2.5:7b",
        host="http://localhost:11434",
        duration_seconds=60.0,
        total_tokens=5000,
        total_requests=20,
        total_failed=1,
        peak_tps=45.2,
        saturation_concurrency=4,
        gpu_name="RTX 5070",
        vram_total_mb=12288.0,
        vram_peak_mb=8500.0,
        phases=[
            ConcurrencyResult(
                concurrency=1, requests=5, total_output_tokens=1000,
                aggregate_tps=20.0, avg_request_tps=20.0,
                avg_latency_ms=200.0, p95_latency_ms=250.0,
            ),
            ConcurrencyResult(
                concurrency=4, requests=15, total_output_tokens=4000,
                aggregate_tps=45.2, avg_request_tps=12.0,
                avg_latency_ms=800.0, p95_latency_ms=1200.0,
            ),
        ],
    )
    repo.save_stress_result(sr)

    rows = repo.get_stress_results()
    assert len(rows) == 1
    r = rows[0]
    assert r["model"] == "qwen2.5:7b"
    assert r["peak_tps"] == 45.2
    assert r["saturation_concurrency"] == 4
    assert r["gpu_name"] == "RTX 5070"
    assert r["total_phases"] == 2
    repo.close()


def test_get_stress_results_multiple_models():
    from atomics.stress import StressResult

    repo = _tmp_repo()

    for model in ["qwen2.5:1.5b", "qwen2.5:7b", "mistral:7b"]:
        sr = StressResult(
            model=model, host="http://localhost:11434",
            peak_tps=30.0, saturation_concurrency=2,
            duration_seconds=30.0, total_tokens=1000, total_requests=10,
        )
        repo.save_stress_result(sr)

    rows = repo.get_stress_results()
    assert len(rows) == 3
    models = [r["model"] for r in rows]
    assert "qwen2.5:1.5b" in models
    assert "mistral:7b" in models
    repo.close()


def test_get_stress_results_by_model():
    from atomics.stress import StressResult

    repo = _tmp_repo()

    for model in ["qwen2.5:1.5b", "qwen2.5:7b"]:
        sr = StressResult(
            model=model, host="http://localhost:11434",
            peak_tps=30.0, saturation_concurrency=2,
            duration_seconds=30.0, total_tokens=1000, total_requests=10,
        )
        repo.save_stress_result(sr)

    rows = repo.get_stress_results(model="qwen2.5:7b")
    assert len(rows) == 1
    assert rows[0]["model"] == "qwen2.5:7b"
    repo.close()


# ── sweep_results ─────────────────────────────────────────────────────────────

def _mock_sweep_result(model: str = "qwen2.5:7b", quality: float = 0.9,
                       provider: str = "ollama") -> object:
    from types import SimpleNamespace
    return SimpleNamespace(
        model=model,
        provider=provider,
        overall_quality=quality,
        avg_latency_ms=1500.0,
        total_tokens=3000,
        total_cost_usd=0.0,
        fixtures_run=6,
    )


def test_save_sweep_result_basic():
    repo = _tmp_repo()
    sr = _mock_sweep_result()
    repo.save_sweep_result(sr)
    rows = repo.get_sweep_results()
    assert len(rows) == 1
    assert rows[0]["model"] == "qwen2.5:7b"
    assert rows[0]["quality"] == 0.9
    assert rows[0]["fixtures_run"] == 6
    repo.close()


def test_save_sweep_result_stores_provider():
    repo = _tmp_repo()
    repo.save_sweep_result(_mock_sweep_result(provider="openai"))
    rows = repo.get_sweep_results()
    assert rows[0]["provider"] == "openai"
    repo.close()


def test_get_sweep_results_by_model():
    repo = _tmp_repo()
    for model in ["qwen2.5:7b", "gpt-4o-mini"]:
        repo.save_sweep_result(_mock_sweep_result(model=model))
    rows = repo.get_sweep_results(model="gpt-4o-mini")
    assert len(rows) == 1
    assert rows[0]["model"] == "gpt-4o-mini"
    repo.close()


def test_sweep_results_table_exists(tmp_path):
    from atomics.storage.schema import init_db
    conn = init_db(tmp_path / "db.sqlite")
    conn.execute(
        "INSERT INTO sweep_results "
        "(result_id, model, provider, quality, avg_latency_ms, total_tokens, "
        "total_cost_usd, fixtures_run, timestamp) "
        "VALUES ('r1','test','ollama',0.9,1000.0,100,0.0,5,'2026-01-01')"
    )
    conn.commit()
    row = conn.execute("SELECT model FROM sweep_results WHERE result_id='r1'").fetchone()
    assert row[0] == "test"
    conn.close()


def _evaluation_record(
    *,
    run_id: str = "eval-run",
    suite: str = "refusal",
    fixture_id: str = "fixture-1",
    status: str = "complete",
    score: float | None = 1.0,
    total_tokens: int = 15,
    error_message: str = "",
) -> EvaluationResultRecord:
    return EvaluationResultRecord(
        run_id=run_id,
        suite=suite,
        fixture_id=fixture_id,
        status=status,
        generation_status="completed",
        judge_status="scored" if score is not None else "provider_error",
        latency_ms=25.0,
        input_tokens=10,
        output_tokens=total_tokens - 10,
        total_tokens=total_tokens,
        result_json={"id": fixture_id, "status": status},
        score=score,
        estimated_cost_usd=0.02,
        attempt_count=1,
        judge_failures=0 if score is not None else 1,
        provider="ollama",
        model="qwen",
        error_message=error_message,
    )


def test_evaluation_results_table_exists_with_foreign_key(tmp_path):
    conn = init_db(tmp_path / "evaluation.db")
    table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='evaluation_results'"
    ).fetchone()
    foreign_keys = conn.execute(
        "PRAGMA foreign_key_list(evaluation_results)"
    ).fetchall()

    assert table is not None
    assert any(row["table"] == "runs" for row in foreign_keys)
    conn.close()


def test_save_evaluation_result_upserts_logical_fixture(tmp_path):
    repo = MetricsRepository(tmp_path / "metrics.db")
    repo.create_run("eval-run", tier="refusal", provider="ollama", model="qwen")
    repo.save_evaluation_result(
        _evaluation_record(status="infrastructure_invalid", score=None)
    )
    repo.save_evaluation_result(_evaluation_record())

    rows = repo.get_evaluation_results(run_id="eval-run", suite="refusal")

    assert len(rows) == 1
    assert rows[0]["status"] == "complete"
    assert rows[0]["score"] == 1.0
    assert rows[0]["result_json"] == {"id": "fixture-1", "status": "complete"}
    repo.close()


def test_get_evaluation_results_filters_suite_and_orders_timestamp(tmp_path):
    repo = MetricsRepository(tmp_path / "metrics.db")
    repo.create_run("eval-run", tier="eval", provider="ollama", model="qwen")
    repo.save_evaluation_result(_evaluation_record(fixture_id="r1"))
    repo.save_evaluation_result(
        _evaluation_record(suite="codereview", fixture_id="c1")
    )

    rows = repo.get_evaluation_results(run_id="eval-run", suite="codereview")

    assert [row["fixture_id"] for row in rows] == ["c1"]
    repo.close()


def test_get_evaluation_results_can_filter_without_run_id(tmp_path):
    repo = MetricsRepository(tmp_path / "metrics.db")
    for run_id in ("run-a", "run-b"):
        repo.create_run(run_id, tier="refusal", provider="ollama", model="qwen")
        repo.save_evaluation_result(_evaluation_record(run_id=run_id))

    rows = repo.get_evaluation_results(suite="refusal")

    assert {row["run_id"] for row in rows} == {"run-a", "run-b"}
    repo.close()


def test_save_evaluation_result_sanitizes_error(tmp_path):
    repo = MetricsRepository(tmp_path / "metrics.db")
    repo.create_run("eval-run", tier="refusal", provider="ollama", model="qwen")
    repo.save_evaluation_result(
        _evaluation_record(
            status="infrastructure_invalid",
            score=None,
            error_message="x-api-key: secret-value",
        )
    )

    row = repo.get_evaluation_results(run_id="eval-run")[0]

    assert "secret-value" not in row["error_message"]
    assert "[REDACTED]" in row["error_message"]
    repo.close()


def test_complete_evaluation_run_rolls_up_honest_counts(tmp_path):
    repo = MetricsRepository(tmp_path / "metrics.db")
    repo.create_run("eval-run", tier="refusal", provider="ollama", model="qwen")
    repo.save_evaluation_result(
        _evaluation_record(fixture_id="complete", total_tokens=15)
    )
    repo.save_evaluation_result(
        _evaluation_record(
            fixture_id="failed",
            status="infrastructure_invalid",
            score=None,
            total_tokens=15,
        )
    )

    summary = repo.complete_evaluation_run("eval-run")

    assert summary.total_tasks == 2
    assert summary.successful_tasks == 1
    assert summary.failed_tasks == 1
    assert summary.total_tokens == 30
    repo.close()


def test_complete_evaluation_run_handles_zero_rows(tmp_path):
    repo = MetricsRepository(tmp_path / "metrics.db")
    repo.create_run("eval-run", tier="refusal", provider="ollama", model="qwen")

    summary = repo.complete_evaluation_run("eval-run")

    assert summary.total_tasks == 0
    assert summary.successful_tasks == 0
    assert summary.failed_tasks == 0
    repo.close()
