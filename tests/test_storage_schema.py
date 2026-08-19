"""Tests for schema initialization and non-destructive column reconciliation.

`init_db` used to respond to a SCHEMA_VERSION bump by backing the database up
and dropping every table. That is far too blunt: a type change or a new
constraint would discard run history. These tests cover the in-place path —
nullable columns via ALTER, everything else via a per-table rebuild that
copies rows.
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


def test_type_change_rebuilds_the_table_and_keeps_rows(
    tmp_path: Path, monkeypatch
) -> None:
    """INTEGER → TEXT on a populated column must not wipe the database."""
    from atomics.storage import schema

    db_path = tmp_path / "typed.db"
    first = init_db(db_path)
    first.execute(
        "INSERT INTO runs (run_id, started_at, total_tasks) VALUES (?, ?, ?)",
        ("keep-me", "2026-01-01T00:00:00+00:00", 7),
    )
    first.commit()
    first.close()

    patched = schema.SCHEMA_SQL.replace(
        "total_tasks     INTEGER DEFAULT 0",
        "total_tasks     TEXT DEFAULT '0'",
        1,
    )
    monkeypatch.setattr(schema, "SCHEMA_SQL", patched)
    monkeypatch.setattr(schema, "SCHEMA_VERSION", schema.SCHEMA_VERSION + 1)

    conn = init_db(db_path)
    try:
        row = conn.execute(
            "SELECT run_id, total_tasks FROM runs WHERE run_id = ?",
            ("keep-me",),
        ).fetchone()
        assert row[0] == "keep-me"
        assert str(row[1]) == "7"
        col_type = {
            info[1]: info[2]
            for info in conn.execute("PRAGMA table_info(runs)")
        }
        assert col_type["total_tasks"].upper() == "TEXT"
    finally:
        conn.close()


def test_dropped_column_rebuilds_and_keeps_other_values(
    tmp_path: Path, monkeypatch
) -> None:
    from atomics.storage import schema

    db_path = tmp_path / "drop-col.db"
    first = init_db(db_path)
    first.execute(
        "INSERT INTO runs (run_id, started_at, model) VALUES (?, ?, ?)",
        ("keep-me", "2026-01-01T00:00:00+00:00", "qwen3:14b"),
    )
    first.commit()
    first.close()

    patched = schema.SCHEMA_SQL.replace(
            "    avg_latency_ms  REAL DEFAULT 0.0,\n",
            "",
            1,
        )
    monkeypatch.setattr(schema, "SCHEMA_SQL", patched)
    monkeypatch.setattr(schema, "SCHEMA_VERSION", schema.SCHEMA_VERSION + 1)

    conn = init_db(db_path)
    try:
        names = _column_names(conn, "runs")
        assert "avg_latency_ms" not in names
        row = conn.execute(
            "SELECT run_id, model FROM runs WHERE run_id = ?", ("keep-me",)
        ).fetchone()
        assert tuple(row) == ("keep-me", "qwen3:14b")
    finally:
        conn.close()


def test_not_null_column_with_default_is_added_without_wipe(
    tmp_path: Path, monkeypatch
) -> None:
    from atomics.storage import schema

    db_path = tmp_path / "notnull.db"
    first = init_db(db_path)
    first.execute(
        "INSERT INTO runs (run_id, started_at) VALUES (?, ?)",
        ("keep-me", "2026-01-01T00:00:00+00:00"),
    )
    first.commit()
    first.close()

    patched = schema.SCHEMA_SQL.replace(
        "    pass_count      INTEGER DEFAULT 1\n",
        "    pass_count      INTEGER DEFAULT 1,\n"
        "    origin          TEXT NOT NULL DEFAULT 'local'\n",
        1,
    )
    monkeypatch.setattr(schema, "SCHEMA_SQL", patched)

    conn = init_db(db_path)
    try:
        names = _column_names(conn, "runs")
        assert "origin" in names
        value = conn.execute(
            "SELECT origin FROM runs WHERE run_id = ?", ("keep-me",)
        ).fetchone()[0]
        assert value == "local"
        assert [row[0] for row in conn.execute("SELECT run_id FROM runs")] == [
            "keep-me"
        ]
    finally:
        conn.close()
