"""Repository for persisting and querying run/task metrics."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from atomics.models import RunSummary, TaskResult, TaskStatus
from atomics.storage.schema import init_db


class MetricsRepository:
    def __init__(self, db_path: Path) -> None:
        self._conn = init_db(db_path)

    def close(self) -> None:
        self._conn.close()

    # ── Runs ──────────────────────────────────────────────

    def create_run(self, run_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO runs (run_id, started_at) VALUES (?, ?)",
            (run_id, now),
        )
        self._conn.commit()

    def complete_run(self, run_id: str) -> RunSummary:
        now = datetime.now(timezone.utc).isoformat()
        rows = self._conn.execute(
            """
            SELECT
                COUNT(*) as total,
                COALESCE(SUM(CASE WHEN status = ? THEN 1 ELSE 0 END), 0) as success,
                COALESCE(SUM(CASE WHEN status = ? THEN 1 ELSE 0 END), 0) as failed,
                COALESCE(SUM(input_tokens), 0) as inp,
                COALESCE(SUM(output_tokens), 0) as outp,
                COALESCE(SUM(total_tokens), 0) as tot,
                COALESCE(SUM(estimated_cost_usd), 0.0) as cost,
                COALESCE(AVG(latency_ms), 0.0) as avg_lat
            FROM task_results WHERE run_id = ?
            """,
            (TaskStatus.SUCCESS.value, TaskStatus.FAILED.value, run_id),
        ).fetchone()

        self._conn.execute(
            """
            UPDATE runs SET
                completed_at = ?,
                total_tasks = ?, successful_tasks = ?, failed_tasks = ?,
                total_input_tokens = ?, total_output_tokens = ?, total_tokens = ?,
                total_cost_usd = ?, avg_latency_ms = ?
            WHERE run_id = ?
            """,
            (now, rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7], run_id),
        )
        self._conn.commit()

        return RunSummary(
            run_id=run_id,
            started_at=datetime.fromisoformat(
                self._conn.execute(
                    "SELECT started_at FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()[0]
            ),
            completed_at=datetime.fromisoformat(now),
            total_tasks=rows[0],
            successful_tasks=rows[1],
            failed_tasks=rows[2],
            total_input_tokens=rows[3],
            total_output_tokens=rows[4],
            total_tokens=rows[5],
            total_cost_usd=rows[6],
            avg_latency_ms=rows[7],
        )

    # ── Task results ──────────────────────────────────────

    def save_task_result(self, result: TaskResult) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO task_results (
                task_id, run_id, category, task_name, provider, model, status,
                prompt, response, input_tokens, output_tokens, total_tokens,
                latency_ms, estimated_cost_usd, error_class, error_message,
                started_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.task_id,
                result.run_id,
                result.category.value,
                result.task_name,
                result.provider,
                result.model,
                result.status.value,
                result.prompt,
                result.response,
                result.input_tokens,
                result.output_tokens,
                result.total_tokens,
                result.latency_ms,
                result.estimated_cost_usd,
                result.error_class,
                result.error_message,
                result.started_at.isoformat(),
                result.completed_at.isoformat() if result.completed_at else None,
            ),
        )
        self._conn.commit()

    # ── Queries ───────────────────────────────────────────

    def get_recent_runs(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_run_tasks(self, run_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM task_results WHERE run_id = ? ORDER BY started_at", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_token_usage_by_hour(self, hours: int = 24) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT
                strftime('%Y-%m-%d %H:00', started_at) as hour,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(estimated_cost_usd) as cost,
                COUNT(*) as task_count
            FROM task_results
            WHERE started_at >= datetime('now', ? || ' hours')
            GROUP BY hour ORDER BY hour
            """,
            (f"-{hours}",),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_usage_by_category(self) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT
                category,
                COUNT(*) as task_count,
                SUM(total_tokens) as total_tokens,
                SUM(estimated_cost_usd) as total_cost,
                AVG(latency_ms) as avg_latency
            FROM task_results
            GROUP BY category ORDER BY total_tokens DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def get_hourly_token_rate(self) -> float:
        """Tokens consumed in the last complete hour."""
        row = self._conn.execute(
            """
            SELECT COALESCE(SUM(total_tokens), 0)
            FROM task_results
            WHERE started_at >= datetime('now', '-1 hour')
            """
        ).fetchone()
        return float(row[0])
