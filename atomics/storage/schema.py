"""SQLite schema definition and migration."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger("atomics.schema")

SCHEMA_VERSION = 2

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS runs (
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
    total_cost_usd  REAL DEFAULT 0.0,
    avg_latency_ms  REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS task_results (
    task_id         TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    category        TEXT NOT NULL,
    task_name       TEXT NOT NULL,
    provider        TEXT NOT NULL,
    model           TEXT NOT NULL,
    status          TEXT NOT NULL,
    prompt          TEXT DEFAULT '',
    response        TEXT DEFAULT '',
    input_tokens    INTEGER DEFAULT 0,
    output_tokens   INTEGER DEFAULT 0,
    total_tokens    INTEGER DEFAULT 0,
    latency_ms      REAL DEFAULT 0.0,
    estimated_cost_usd REAL DEFAULT 0.0,
    error_class     TEXT DEFAULT '',
    error_message   TEXT DEFAULT '',
    started_at      TEXT NOT NULL,
    completed_at    TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS schedules (
    schedule_id     TEXT PRIMARY KEY,
    format          TEXT NOT NULL,
    tier            TEXT NOT NULL,
    provider        TEXT NOT NULL,
    model           TEXT,
    interval_minutes INTEGER NOT NULL,
    max_iterations  INTEGER NOT NULL,
    installed_at    TEXT NOT NULL,
    last_run_at     TEXT,
    last_status     TEXT
);

CREATE INDEX IF NOT EXISTS idx_task_results_run_id ON task_results(run_id);
CREATE INDEX IF NOT EXISTS idx_task_results_category ON task_results(category);
CREATE INDEX IF NOT EXISTS idx_task_results_started_at ON task_results(started_at);
CREATE INDEX IF NOT EXISTS idx_runs_provider ON runs(provider);
CREATE INDEX IF NOT EXISTS idx_runs_tier ON runs(tier);
CREATE INDEX IF NOT EXISTS idx_runs_trigger ON runs(trigger);
"""


def _get_schema_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        return row[0] if row[0] is not None else 0
    except sqlite3.OperationalError:
        return 0


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize the database, creating tables if needed.

    On schema version mismatch the DB is wiped and recreated (fresh-start
    policy while the project is pre-1.0).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    current = _get_schema_version(conn)
    if current != 0 and current < SCHEMA_VERSION:
        logger.info(
            "Schema version %d → %d: dropping all tables for fresh start.",
            current,
            SCHEMA_VERSION,
        )
        conn.executescript(
            "DROP TABLE IF EXISTS task_results;"
            "DROP TABLE IF EXISTS runs;"
            "DROP TABLE IF EXISTS schedules;"
            "DROP TABLE IF EXISTS schema_version;"
        )

    conn.executescript(SCHEMA_SQL)
    conn.execute(
        "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
        (SCHEMA_VERSION,),
    )
    conn.commit()
    return conn
