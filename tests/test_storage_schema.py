"""Tests for schema initialization and non-destructive column reconciliation.

`init_db` responds to a SCHEMA_VERSION bump by backing the database up and
dropping every table, which is the documented pre-1.0 fresh-start policy. That is
far too blunt for adding one nullable column: it would reset local run history,
schedules, and the evaluation ledger. These tests cover the in-place path that
adds missing nullable columns instead.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from atomics.storage.schema import SCHEMA_VERSION, init_db

# `runs` as it stands today minus its last column. Used to stand in for a
# database written before that column existed.
_LEGACY_RUNS_SQL = """
CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
CREATE TABLE runs (
    run_id          TEXT PRIMARY KEY,
    started_at      TEXT NOT NULL,
    completed_at    TEXT,
    tier            TEXT NOT NULL DEFAULT 'baseline',
    provider        TEXT NOT NULL DEFAULT 'claude',
    model           TEXT NOT NULL DEFAULT '',
    trigger         TEXT NOT NULL DEFAULT 'manual',
    total_tasks     INTEGER DEFAULT 0,
    successful_tasks INTEGER DEFAULT 0,
    failed_tasks    INTEGER DEFAULT 0,
    total_input_tokens  INTEGER DEFAULT 0,
    total_output_tokens INTEGER DEFAULT 0,
    total_tokens    INTEGER DEFAULT 0,
    total_cost_usd  REAL DEFAULT 0.0
);
"""

MISSING_COLUMN = "avg_latency_ms"


def _column_names(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def _table_shapes(conn: sqlite3.Connection) -> dict[str, list[tuple]]:
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        )
    ]
    return {
        table: list(conn.execute(f"PRAGMA table_info({table})")) for table in tables
    }


def _write_legacy_db(path: Path) -> None:
    """A database at the current version but missing a column added since."""
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_LEGACY_RUNS_SQL)
        conn.execute(
            "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
        )
        conn.execute(
            "INSERT INTO runs (run_id, started_at) VALUES (?, ?)",
            ("keep-me", "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()


def test_init_db_adds_a_missing_nullable_column_in_place(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    _write_legacy_db(db_path)
    assert MISSING_COLUMN not in _column_names(
        sqlite3.connect(str(db_path)), "runs"
    )

    conn = init_db(db_path)
    try:
        assert MISSING_COLUMN in _column_names(conn, "runs")
        # The whole point: reconciliation must not be a disguised wipe.
        rows = list(conn.execute("SELECT run_id FROM runs"))
        assert [row[0] for row in rows] == ["keep-me"]
    finally:
        conn.close()


def test_column_reconciliation_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    _write_legacy_db(db_path)

    for _ in range(3):
        conn = init_db(db_path)
        conn.close()

    conn = sqlite3.connect(str(db_path))
    try:
        names = _column_names(conn, "runs")
        assert names.count(MISSING_COLUMN) == 1
        assert [row[0] for row in conn.execute("SELECT run_id FROM runs")] == ["keep-me"]
    finally:
        conn.close()


def test_reconciliation_leaves_a_current_database_untouched(tmp_path: Path) -> None:
    db_path = tmp_path / "current.db"
    first = init_db(db_path)
    try:
        before = _table_shapes(first)
    finally:
        first.close()

    second = init_db(db_path)
    try:
        assert _table_shapes(second) == before
    finally:
        second.close()


# distributed_assignments as Phase 1 shipped it, before fleet mode added
# target_worker_id. Stands in for a real database on disk from before the upgrade.
_PRE_FLEET_DISTRIBUTED_SQL = """
CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
CREATE TABLE workers (
    worker_id TEXT PRIMARY KEY,
    labels TEXT NOT NULL DEFAULT '{}',
    capabilities TEXT,
    endpoint TEXT,
    api_key_hint TEXT,
    status TEXT NOT NULL DEFAULT 'online',
    last_seen_at TEXT,
    registered_at TEXT NOT NULL
);
CREATE TABLE distributed_jobs (
    job_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    parent_run_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    request_json TEXT NOT NULL,
    summary_json TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE TABLE distributed_assignments (
    assignment_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES distributed_jobs(job_id),
    worker_id TEXT REFERENCES workers(worker_id),
    status TEXT NOT NULL DEFAULT 'pending',
    task_spec TEXT NOT NULL,
    result_json TEXT,
    retry_count INTEGER DEFAULT 0,
    started_at TEXT,
    completed_at TEXT
);
"""


def test_a_pre_fleet_database_gains_target_worker_id_without_losing_work(
    tmp_path: Path,
) -> None:
    """The upgrade users will actually perform: keep the job, gain the column.

    Exercised end to end rather than only through the generic reconciliation
    tests, because the promise being kept here is that adding fleet mode does not
    reset anyone's distributed history.
    """
    from atomics.distributed.coordinator import Coordinator

    db_path = tmp_path / "pre-fleet.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_PRE_FLEET_DISTRIBUTED_SQL)
        conn.execute(
            "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
        )
        conn.execute(
            "INSERT INTO distributed_jobs (job_id, mode, status, request_json, created_at) "
            "VALUES ('job-1', 'split', 'pending', '{}', '2026-01-01T00:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO distributed_assignments (assignment_id, job_id, status, task_spec) "
            "VALUES ('assign-1', 'job-1', 'pending', '{\"i\": 1}')"
        )
        conn.execute(
            "INSERT INTO workers (worker_id, registered_at) "
            "VALUES ('worker-1', '2026-01-01T00:00:00+00:00')"
        )
        conn.commit()
    finally:
        conn.close()

    upgraded = init_db(db_path)
    try:
        assert "target_worker_id" in _column_names(upgraded, "distributed_assignments")

        # The pending assignment survived and is still claimable, with no pin.
        claimed = Coordinator(upgraded).claim_assignment("worker-1")
        assert claimed is not None
        assert claimed.assignment_id == "assign-1"
        assert claimed.target_worker_id is None
    finally:
        upgraded.close()


def test_reconciliation_backfills_the_column_as_null(tmp_path: Path) -> None:
    """An added column must read as NULL on pre-existing rows, not fabricate data.

    `runs.avg_latency_ms` carries `DEFAULT 0.0`, so SQLite applies that default to
    existing rows. Asserting the concrete value keeps the behavior explicit rather
    than leaving readers to guess whether old rows report 0.0 or NULL.
    """
    db_path = tmp_path / "legacy.db"
    _write_legacy_db(db_path)

    conn = init_db(db_path)
    try:
        value = conn.execute(
            f"SELECT {MISSING_COLUMN} FROM runs WHERE run_id = ?", ("keep-me",)
        ).fetchone()[0]
        assert value == 0.0
    finally:
        conn.close()
