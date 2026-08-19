"""SQLite schema definition and migration."""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("atomics.schema")

SCHEMA_VERSION = 21

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
    avg_latency_ms  REAL DEFAULT 0.0,
    pass_count      INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    result_id               TEXT PRIMARY KEY,
    run_id                  TEXT NOT NULL,
    suite                   TEXT NOT NULL,
    fixture_id              TEXT NOT NULL,
    status                  TEXT NOT NULL DEFAULT '',
    score                   REAL DEFAULT NULL,
    generation_status       TEXT NOT NULL DEFAULT '',
    judge_status            TEXT NOT NULL DEFAULT '',
    latency_ms              REAL DEFAULT 0.0,
    estimated_cost_usd      REAL DEFAULT 0.0,
    input_tokens            INTEGER NOT NULL DEFAULT 0,
    output_tokens           INTEGER NOT NULL DEFAULT 0,
    total_tokens            INTEGER NOT NULL DEFAULT 0,
    thinking_tokens         INTEGER NOT NULL DEFAULT 0,
    attempt_count           INTEGER NOT NULL DEFAULT 0,
    generation_failures     INTEGER NOT NULL DEFAULT 0,
    infrastructure_failures INTEGER NOT NULL DEFAULT 0,
    judge_failures          INTEGER NOT NULL DEFAULT 0,
    parse_failed            INTEGER NOT NULL DEFAULT 0,
    provider                TEXT NOT NULL DEFAULT '',
    model                   TEXT NOT NULL DEFAULT '',
    error_class             TEXT NOT NULL DEFAULT '',
    error_message           TEXT NOT NULL DEFAULT '',
    result_json             TEXT NOT NULL DEFAULT '{}',
    timestamp               TEXT NOT NULL,
    judge_agreement         REAL DEFAULT NULL,
    UNIQUE (run_id, suite, fixture_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS judge_agreement_results (
    result_id     TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    suite         TEXT NOT NULL,
    fixture_id    TEXT NOT NULL,
    votes_json    TEXT NOT NULL DEFAULT '{}',
    agreement     REAL DEFAULT NULL,
    flipped       INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    UNIQUE (run_id, suite, fixture_id)
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
    timestamp           TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT '',
    generation_status   TEXT NOT NULL DEFAULT '',
    judge_status        TEXT NOT NULL DEFAULT '',
    attempt_count       INTEGER NOT NULL DEFAULT 0,
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    total_tokens        INTEGER NOT NULL DEFAULT 0,
    attempts_json       TEXT NOT NULL DEFAULT '[]',
    run_scores_json     TEXT NOT NULL DEFAULT '[]',
    generation_failures INTEGER NOT NULL DEFAULT 0,
    infrastructure_failures INTEGER NOT NULL DEFAULT 0,
    judge_failures      INTEGER NOT NULL DEFAULT 0,
    parse_failed        INTEGER NOT NULL DEFAULT 0,
    error_class         TEXT NOT NULL DEFAULT '',
    error_message       TEXT NOT NULL DEFAULT '',
    UNIQUE (run_id, fixture_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
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
CREATE INDEX IF NOT EXISTS idx_evaluation_results_run
    ON evaluation_results(run_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_results_suite
    ON evaluation_results(suite);
CREATE INDEX IF NOT EXISTS idx_evaluation_results_timestamp
    ON evaluation_results(timestamp);
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

CREATE TABLE IF NOT EXISTS workers (
    worker_id TEXT PRIMARY KEY,
    labels TEXT NOT NULL,
    capabilities TEXT,
    endpoint TEXT,
    api_key_hint TEXT,
    status TEXT NOT NULL DEFAULT 'online',
    last_seen_at TEXT,
    registered_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS distributed_jobs (
    job_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    parent_run_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    request_json TEXT NOT NULL,
    summary_json TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS distributed_assignments (
    assignment_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES distributed_jobs(job_id),
    worker_id TEXT REFERENCES workers(worker_id),
    -- NULL means any worker may claim this (split mode). A value pins the
    -- assignment to one worker and only that worker (fleet mode), so a
    -- per-host comparison cannot be silently satisfied by another host.
    target_worker_id TEXT REFERENCES workers(worker_id),
    status TEXT NOT NULL DEFAULT 'pending',
    task_spec TEXT NOT NULL,
    result_json TEXT,
    retry_count INTEGER DEFAULT 0,
    started_at TEXT,
    completed_at TEXT
);
"""

RESET_SQL = """
DROP TABLE IF EXISTS task_results;
DROP TABLE IF EXISTS adversarial_results;
DROP TABLE IF EXISTS evaluation_results;
DROP TABLE IF EXISTS probe_results;
DROP TABLE IF EXISTS stress_results;
DROP TABLE IF EXISTS sweep_results;
DROP TABLE IF EXISTS scenario_results;
DROP TABLE IF EXISTS soak_results;
DROP TABLE IF EXISTS baselines;
DROP TABLE IF EXISTS archreview_results;
DROP TABLE IF EXISTS labcompare_results;
DROP TABLE IF EXISTS distributed_assignments;
DROP TABLE IF EXISTS distributed_jobs;
DROP TABLE IF EXISTS workers;
DROP TABLE IF EXISTS runs;
DROP TABLE IF EXISTS schedules;
DROP TABLE IF EXISTS schema_version;
"""


def _get_schema_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        return row[0] if row[0] is not None else 0
    except sqlite3.OperationalError:
        return 0


def _backup_before_wipe(conn: sqlite3.Connection, db_path: Path, current: int) -> Path:
    """WAL-safe snapshot of the DB before an in-place schema migration.

    The caller holds a write lock on ``conn``. A separate read connection avoids
    the online backup API deadlocking on that connection while the lock prevents
    concurrent commits from racing the snapshot.
    """
    if not conn.in_transaction:
        raise sqlite3.ProgrammingError("migration backup requires an active lock")
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    backup_path = db_path.with_name(f"{db_path.name}.v{current}.{stamp}.bak")
    source_conn = sqlite3.connect(str(db_path))
    try:
        backup_conn = sqlite3.connect(str(backup_path))
        try:
            with backup_conn:
                source_conn.backup(backup_conn)
        finally:
            backup_conn.close()
    finally:
        source_conn.close()
    return backup_path


def _scratch_schema() -> tuple[dict[str, list[tuple]], dict[str, str]]:
    """Columns and CREATE TABLE SQL SCHEMA_SQL would produce.

    Parsing the DDL by hand would mean maintaining a second description of the
    schema alongside the first, free to drift from it. Letting SQLite build the
    schema and report on it keeps one source of truth.
    """
    scratch = sqlite3.connect(":memory:")
    try:
        scratch.executescript(SCHEMA_SQL)
        tables = [
            row[0] for row in scratch.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        ]
        columns = {table: list(scratch.execute(f"PRAGMA table_info({table})")) for table in tables}
        ddl: dict[str, str] = {}
        for table in tables:
            row = scratch.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            if row and row[0]:
                ddl[table] = row[0]
        return columns, ddl
    finally:
        scratch.close()


def _expected_table_columns() -> dict[str, list[tuple]]:
    columns, _ddl = _scratch_schema()
    return columns


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _rename_create_table(sql: str, old: str, new: str) -> str:
    for prefix in (f"CREATE TABLE {old}", f'CREATE TABLE "{old}"'):
        if sql.startswith(prefix):
            return f"CREATE TABLE {new}" + sql[len(prefix) :]
    raise sqlite3.OperationalError(f"unrecognized CREATE TABLE for {old}")


def _same_column(existing: tuple, expected: tuple) -> bool:
    """Type, NOT NULL, and primary-key flag. Defaults may differ without a rebuild."""
    return (
        (existing[2] or "").upper() == (expected[2] or "").upper()
        and existing[3] == expected[3]
        and existing[5] == expected[5]
    )


def _can_alter_add(expected: tuple) -> bool:
    """SQLite can ADD a column that is not a PRIMARY KEY.

    NOT NULL is allowed when a DEFAULT is present, which is how SCHEMA_SQL
    writes almost every new column.
    """
    _cid, _name, _col_type, notnull, default, pk = expected
    if pk:
        return False
    if notnull and default is None:
        return False
    return True


def _rebuild_needed(existing: list[tuple], expected: list[tuple]) -> bool:
    exist_by = {row[1]: row for row in existing}
    exp_by = {row[1]: row for row in expected}
    if set(exist_by) - set(exp_by):
        return True
    for name, exp in exp_by.items():
        if name not in exist_by:
            if not _can_alter_add(exp):
                return True
            continue
        if not _same_column(exist_by[name], exp):
            return True
    return False


def _rebuild_table(
    conn: sqlite3.Connection,
    table: str,
    create_sql: str,
    expected: list[tuple],
    existing: list[tuple],
) -> None:
    """Copy shared columns into a new table that matches SCHEMA_SQL."""
    common = [row[1] for row in expected if row[1] in {item[1] for item in existing}]
    if not common:
        raise sqlite3.OperationalError(f"cannot rebuild {table}: no shared columns")
    tmp = f"{table}__new"
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute(_rename_create_table(create_sql, table, tmp))
    cols = ", ".join(_ident(name) for name in common)
    conn.execute(f"INSERT INTO {_ident(tmp)} ({cols}) SELECT {cols} FROM {_ident(table)}")
    conn.execute(f"DROP TABLE {_ident(table)}")
    conn.execute(f"ALTER TABLE {_ident(tmp)} RENAME TO {_ident(table)}")
    conn.execute("PRAGMA foreign_keys=ON")


def _add_missing_columns(
    conn: sqlite3.Connection, table: str, expected: list[tuple], existing_names: set[str]
) -> list[str]:
    added: list[str] = []
    for expected_col in expected:
        _cid, name, col_type, notnull, default, pk = expected_col
        if name in existing_names:
            continue
        if not _can_alter_add(expected_col):
            logger.warning(
                "Cannot add %s.%s in place (PRIMARY KEY or NOT NULL without "
                "DEFAULT); rebuilding the table instead.",
                table,
                name,
            )
            continue
        clause = f"{name} {col_type}".strip()
        if notnull:
            clause += " NOT NULL"
        if default is not None:
            clause += f" DEFAULT {default}"
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {clause}")
        added.append(f"{table}.{name}")
    return added


def _reconcile_schema(conn: sqlite3.Connection) -> list[str]:
    """Bring an existing database in line with SCHEMA_SQL without a wipe.

    Missing columns that SQLite can ADD are added in place. A type change, a
    dropped column, a PRIMARY KEY, or a NOT NULL without DEFAULT rebuilds that
    one table and copies the shared columns. Other tables are left alone.
    """
    expected_columns, ddl = _scratch_schema()
    actions: list[str] = []
    rebuilt: list[str] = []
    for table, expected in expected_columns.items():
        existing = list(conn.execute(f"PRAGMA table_info({table})"))
        if not existing:
            continue
        existing_names = {row[1] for row in existing}
        if _rebuild_needed(existing, expected):
            create_sql = ddl.get(table)
            if not create_sql:
                raise sqlite3.OperationalError(f"no CREATE TABLE for {table}")
            _rebuild_table(conn, table, create_sql, expected, existing)
            rebuilt.append(table)
            actions.append(f"rebuild:{table}")
            continue
        added = _add_missing_columns(conn, table, expected, existing_names)
        actions.extend(f"add:{name}" for name in added)
    if rebuilt:
        logger.info("Rebuilt tables in place: %s", ", ".join(rebuilt))
    added_names = [item.removeprefix("add:") for item in actions if item.startswith("add:")]
    if added_names:
        logger.info("Added missing columns in place: %s", ", ".join(added_names))
    return actions


def _reconcile_added_columns(conn: sqlite3.Connection) -> list[str]:
    """Backward-compatible name for the add-or-rebuild path."""
    return _reconcile_schema(conn)


def _any_rebuild_needed(conn: sqlite3.Connection) -> bool:
    expected_columns, _ddl = _scratch_schema()
    for table, expected in expected_columns.items():
        existing = list(conn.execute(f"PRAGMA table_info({table})"))
        if existing and _rebuild_needed(existing, expected):
            return True
    return False


def _execute_sql_statements(conn: sqlite3.Connection, script: str) -> None:
    """Execute a SQL script without sqlite3's implicit transaction commits."""
    statement = ""
    for char in script:
        statement += char
        if char == ";" and sqlite3.complete_statement(statement):
            if statement.strip():
                conn.execute(statement)
            statement = ""
    if statement.strip():
        raise sqlite3.OperationalError("incomplete SQL statement")


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize the database, creating tables if needed.

    A schema version bump no longer drops tables. The existing DB is
    snapshotted to a timestamped ``.bak``, then each table is reconciled in
    place: missing columns are added, and a type or constraint change rebuilds
    only that table while copying rows. ``RESET_SQL`` is kept for a manual wipe
    and is not used here.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")

        current = _get_schema_version(conn)
        migration_candidate = current != 0 and current < SCHEMA_VERSION
        conn.execute("BEGIN IMMEDIATE")
        if migration_candidate:
            current = _get_schema_version(conn)
        version_bump = current != 0 and current < SCHEMA_VERSION
        if version_bump or (current != 0 and _any_rebuild_needed(conn)):
            backup_path = _backup_before_wipe(conn, db_path, current)
            logger.warning(
                "Schema version %d → %d: backed up existing DB to %s, then "
                "migrating tables in place.",
                current,
                SCHEMA_VERSION,
                backup_path,
            )

        # Reconcile existing tables before CREATE INDEX runs: an old `runs`
        # table may not yet have `provider`, and SCHEMA_SQL creates that index.
        _reconcile_schema(conn)
        _execute_sql_statements(conn, SCHEMA_SQL)
        conn.execute(
            "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
            (SCHEMA_VERSION,),
        )
        conn.commit()
        return conn
    except Exception:
        conn.rollback()
        conn.close()
        raise
