"""SQLite schema definition and migration."""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("atomics.schema")

SCHEMA_VERSION = 16

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
    suite           TEXT NOT NULL DEFAULT 'eval',
    prompt          TEXT DEFAULT '',
    response        TEXT DEFAULT '',
    input_tokens    INTEGER DEFAULT 0,
    output_tokens   INTEGER DEFAULT 0,
    total_tokens    INTEGER DEFAULT 0,
    thinking_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    latency_ms      REAL DEFAULT 0.0,
    estimated_cost_usd REAL DEFAULT 0.0,
    tokens_per_second REAL DEFAULT NULL,
    tps_basis       TEXT DEFAULT 'wall_clock',
    thinking_enabled INTEGER DEFAULT 0,
    error_class     TEXT DEFAULT '',
    error_message   TEXT DEFAULT '',
    started_at      TEXT NOT NULL,
    completed_at    TEXT,
    accuracy_score  REAL DEFAULT NULL,
    judge_model     TEXT DEFAULT '',
    quality_rationale TEXT DEFAULT '',
    criteria_coverage REAL DEFAULT NULL,
    judge_score_stdev REAL DEFAULT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS adversarial_results (
    result_id           TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL,
    fixture_id          TEXT NOT NULL,
    category            TEXT NOT NULL,
    severity            TEXT NOT NULL,
    provider            TEXT NOT NULL,
    model               TEXT NOT NULL,
    prompt              TEXT DEFAULT '',
    response            TEXT DEFAULT '',
    attack_goal         TEXT DEFAULT '',
    resistance_score    REAL DEFAULT NULL,
    resistance_label    TEXT DEFAULT '',
    judge_model         TEXT DEFAULT '',
    judge_rationale     TEXT DEFAULT '',
    thinking_enabled    INTEGER DEFAULT 0,
    thinking_tokens     INTEGER DEFAULT 0,
    latency_ms          REAL DEFAULT 0.0,
    estimated_cost_usd  REAL DEFAULT 0.0,
    timestamp           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS probe_results (
    result_id           TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL,
    target_name         TEXT NOT NULL,
    artifact_type       TEXT NOT NULL,
    check_id            TEXT NOT NULL,
    score               REAL DEFAULT NULL,
    prev_score          REAL DEFAULT NULL,
    regressed           INTEGER DEFAULT 0,
    provider            TEXT NOT NULL,
    model               TEXT NOT NULL,
    judge_model         TEXT DEFAULT '',
    judge_rationale     TEXT DEFAULT '',
    thinking_enabled    INTEGER DEFAULT 0,
    thinking_tokens     INTEGER DEFAULT 0,
    timestamp           TEXT NOT NULL
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
CREATE INDEX IF NOT EXISTS idx_task_results_suite ON task_results(suite);
CREATE INDEX IF NOT EXISTS idx_task_results_started_at ON task_results(started_at);
CREATE INDEX IF NOT EXISTS idx_runs_provider ON runs(provider);
CREATE INDEX IF NOT EXISTS idx_runs_tier ON runs(tier);
CREATE INDEX IF NOT EXISTS idx_runs_trigger ON runs(trigger);
CREATE TABLE IF NOT EXISTS stress_results (
    result_id               TEXT PRIMARY KEY,
    model                   TEXT NOT NULL,
    host                    TEXT NOT NULL,
    peak_tps                REAL DEFAULT 0.0,
    saturation_concurrency  INTEGER DEFAULT 0,
    duration_seconds        REAL DEFAULT 0.0,
    total_tokens            INTEGER DEFAULT 0,
    total_requests          INTEGER DEFAULT 0,
    total_failed            INTEGER DEFAULT 0,
    total_phases            INTEGER DEFAULT 0,
    gpu_name                TEXT DEFAULT '',
    vram_total_mb           REAL DEFAULT NULL,
    vram_peak_mb            REAL DEFAULT NULL,
    phases_json             TEXT DEFAULT '[]',
    timestamp               TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sweep_results (
    result_id       TEXT PRIMARY KEY,
    model           TEXT NOT NULL,
    provider        TEXT NOT NULL DEFAULT '',
    quality         REAL DEFAULT NULL,
    avg_latency_ms  REAL DEFAULT 0.0,
    total_tokens    INTEGER DEFAULT 0,
    total_cost_usd  REAL DEFAULT 0.0,
    fixtures_run    INTEGER DEFAULT 0,
    timestamp       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scenario_results (
    result_id           TEXT PRIMARY KEY,
    duration_seconds    REAL DEFAULT 0.0,
    total_requests      INTEGER DEFAULT 0,
    total_failed        INTEGER DEFAULT 0,
    workload_count      INTEGER DEFAULT 0,
    max_interference    REAL DEFAULT NULL,
    workloads_json      TEXT DEFAULT '[]',
    interference_json   TEXT DEFAULT '{}',
    baselines_json      TEXT DEFAULT '{}',
    timestamp           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS soak_results (
    result_id               TEXT PRIMARY KEY,
    model                   TEXT NOT NULL,
    host                    TEXT NOT NULL DEFAULT '',
    provider                TEXT NOT NULL DEFAULT 'ollama',
    concurrency             INTEGER DEFAULT 4,
    duration_seconds        REAL DEFAULT 0.0,
    actual_duration_seconds REAL DEFAULT 0.0,
    sample_interval         INTEGER DEFAULT 30,
    total_requests          INTEGER DEFAULT 0,
    total_failed            INTEGER DEFAULT 0,
    total_tokens            INTEGER DEFAULT 0,
    avg_tps                 REAL DEFAULT 0.0,
    peak_tps                REAL DEFAULT 0.0,
    min_tps                 REAL DEFAULT 0.0,
    throughput_drift_pct    REAL DEFAULT 0.0,
    latency_drift_pct       REAL DEFAULT 0.0,
    avg_p95_ms              REAL DEFAULT 0.0,
    vram_start_mb           REAL DEFAULT NULL,
    vram_end_mb             REAL DEFAULT NULL,
    vram_drift_mb           REAL DEFAULT NULL,
    error_rate              REAL DEFAULT 0.0,
    verdict                 TEXT DEFAULT 'STABLE',
    total_cost_usd          REAL DEFAULT 0.0,
    samples_json            TEXT DEFAULT '[]',
    timestamp               TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_adversarial_results_run_id ON adversarial_results(run_id);
CREATE INDEX IF NOT EXISTS idx_probe_results_run_id ON probe_results(run_id);
CREATE INDEX IF NOT EXISTS idx_stress_results_model ON stress_results(model);
CREATE INDEX IF NOT EXISTS idx_sweep_results_model ON sweep_results(model);
CREATE INDEX IF NOT EXISTS idx_scenario_results_timestamp ON scenario_results(timestamp);
CREATE INDEX IF NOT EXISTS idx_soak_results_model ON soak_results(model);
CREATE INDEX IF NOT EXISTS idx_soak_results_verdict ON soak_results(verdict);

CREATE TABLE IF NOT EXISTS baselines (
    baseline_id     TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    suite           TEXT NOT NULL DEFAULT 'soak',
    model           TEXT NOT NULL DEFAULT '',
    host            TEXT NOT NULL DEFAULT '',
    avg_tps         REAL DEFAULT 0.0,
    peak_tps        REAL DEFAULT 0.0,
    avg_p95_ms      REAL DEFAULT 0.0,
    error_rate      REAL DEFAULT 0.0,
    verdict         TEXT DEFAULT 'STABLE',
    concurrency     INTEGER DEFAULT 0,
    notes           TEXT DEFAULT '',
    timestamp       TEXT NOT NULL,
    UNIQUE(name, suite)
);

CREATE INDEX IF NOT EXISTS idx_baselines_name ON baselines(name);
CREATE INDEX IF NOT EXISTS idx_baselines_suite ON baselines(suite);

CREATE TABLE IF NOT EXISTS archreview_results (
    result_id               TEXT PRIMARY KEY,
    run_id                  TEXT NOT NULL,
    repo                    TEXT NOT NULL,
    tier                    TEXT NOT NULL,
    model                   TEXT NOT NULL,
    provider                TEXT NOT NULL,
    round                   INTEGER DEFAULT 1,
    objective_recall        REAL DEFAULT NULL,
    objective_precision     REAL DEFAULT NULL,
    objective_f             REAL DEFAULT NULL,
    judge_score             REAL DEFAULT NULL,
    judge_rematch_recall    REAL DEFAULT NULL,
    finding_count           INTEGER DEFAULT 0,
    parse_failed            INTEGER DEFAULT 0,
    tokens_in               INTEGER DEFAULT 0,
    tokens_out              INTEGER DEFAULT 0,
    cost_usd                REAL DEFAULT 0.0,
    latency_ms              REAL DEFAULT 0.0,
    judge_model             TEXT DEFAULT '',
    pack_hash               TEXT DEFAULT '',
    findings_json           TEXT DEFAULT '[]',
    matched_categories_json TEXT DEFAULT '[]',
    error_class             TEXT DEFAULT '',
    error_message           TEXT DEFAULT '',
    timestamp               TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_archreview_results_run_id ON archreview_results(run_id);
CREATE INDEX IF NOT EXISTS idx_archreview_results_repo ON archreview_results(repo);
CREATE INDEX IF NOT EXISTS idx_archreview_results_model ON archreview_results(model);

CREATE TABLE IF NOT EXISTS labcompare_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    comparison_run_id   TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    host_name           TEXT NOT NULL,
    host_url            TEXT NOT NULL,
    model               TEXT NOT NULL,
    tokens_per_second   REAL,
    latency_ms          REAL,
    prompt_eval_rate    REAL,
    vram_fit_pct        REAL,
    gpu_name            TEXT,
    quality_score       REAL,
    quality_suite       TEXT,
    judge_model         TEXT,
    dimensions          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_labcompare_run ON labcompare_results(comparison_run_id);
"""


def _get_schema_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        return row[0] if row[0] is not None else 0
    except sqlite3.OperationalError:
        return 0


def _backup_before_wipe(conn: sqlite3.Connection, db_path: Path, current: int) -> Path:
    """WAL-safe snapshot of the DB before a destructive schema reset.

    Uses SQLite's online backup API (not a file copy) so in-flight WAL data is
    captured. Returns the backup path so callers/logs can point users to it.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    backup_path = db_path.with_name(f"{db_path.name}.v{current}.{stamp}.bak")
    backup_conn = sqlite3.connect(str(backup_path))
    try:
        with backup_conn:
            conn.backup(backup_conn)
    finally:
        backup_conn.close()
    return backup_path


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize the database, creating tables if needed.

    On a schema version bump the tables are reset (fresh-start policy while the
    project is pre-1.0), but the existing DB is first snapshotted to a
    timestamped ``.bak`` beside it, so historical metrics are recoverable rather
    than silently lost.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    current = _get_schema_version(conn)
    if current != 0 and current < SCHEMA_VERSION:
        try:
            backup_path = _backup_before_wipe(conn, db_path, current)
            logger.warning(
                "Schema version %d → %d: backed up existing DB to %s, then "
                "resetting tables.",
                current, SCHEMA_VERSION, backup_path,
            )
        except sqlite3.Error as exc:
            logger.warning(
                "Schema version %d → %d: backup failed (%s); resetting tables.",
                current, SCHEMA_VERSION, exc,
            )
        conn.executescript(
            "DROP TABLE IF EXISTS task_results;"
            "DROP TABLE IF EXISTS adversarial_results;"
            "DROP TABLE IF EXISTS probe_results;"
            "DROP TABLE IF EXISTS stress_results;"
            "DROP TABLE IF EXISTS sweep_results;"
            "DROP TABLE IF EXISTS scenario_results;"
            "DROP TABLE IF EXISTS soak_results;"
            "DROP TABLE IF EXISTS baselines;"
            "DROP TABLE IF EXISTS archreview_results;"
            "DROP TABLE IF EXISTS labcompare_results;"
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
