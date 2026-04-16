"""Repository for persisting and querying run/task metrics."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from atomics.models import RunSummary, TaskResult, TaskStatus
from atomics.storage.schema import init_db


class MetricsRepository:
    def __init__(self, db_path: Path) -> None:
        self._conn = init_db(db_path)

    def close(self) -> None:
        self._conn.close()

    # ── Runs ──────────────────────────────────────────────

    def create_run(
        self,
        run_id: str,
        *,
        tier: str = "baseline",
        provider: str = "claude",
        model: str = "",
        trigger: str = "manual",
    ) -> None:
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "INSERT INTO runs (run_id, started_at, tier, provider, model, trigger) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, now, tier, provider, model, trigger),
        )
        self._conn.commit()

    def complete_run(self, run_id: str) -> RunSummary:
        now = datetime.now(UTC).isoformat()
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

    def query_task_results(
        self,
        *,
        since_hours: float | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Return task rows for export, newest first."""
        clauses: list[str] = []
        params: list = []
        if since_hours is not None:
            clauses.append("started_at >= datetime('now', ?)")
            params.append(f"-{since_hours} hours")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM task_results {where} ORDER BY started_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ── Provider comparison ───────────────────────────────

    def compare_providers(
        self,
        *,
        since_hours: float | None = None,
        tier: str | None = None,
        category: str | None = None,
        group_by: str = "provider",
    ) -> list[dict]:
        """Aggregate task metrics grouped by provider or model."""
        clauses: list[str] = []
        params: list = []
        if since_hours is not None:
            clauses.append("started_at >= datetime('now', ?)")
            params.append(f"-{since_hours} hours")
        if tier is not None:
            clauses.append(
                "run_id IN (SELECT run_id FROM runs WHERE tier = ?)"
            )
            params.append(tier)
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        col = "provider" if group_by == "provider" else "model"
        other = "model" if group_by == "provider" else "provider"
        sql = f"""
            SELECT
                {col} as group_key,
                GROUP_CONCAT(DISTINCT {other}) as models_used,
                COUNT(*) as task_count,
                COALESCE(SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END), 0) as successes,
                COALESCE(AVG(total_tokens), 0) as avg_tokens,
                COALESCE(AVG(latency_ms), 0) as avg_latency_ms,
                COALESCE(AVG(estimated_cost_usd), 0) as avg_cost_per_task,
                COALESCE(SUM(estimated_cost_usd), 0) as total_cost,
                COALESCE(SUM(total_tokens), 0) as total_tokens
            FROM task_results {where}
            GROUP BY {col}
            ORDER BY avg_cost_per_task ASC
        """
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_runs_by_provider(
        self,
        *,
        since_hours: float | None = None,
    ) -> list[dict]:
        """Aggregate run-level metrics grouped by provider."""
        clauses: list[str] = []
        params: list = []
        if since_hours is not None:
            clauses.append("started_at >= datetime('now', ?)")
            params.append(f"-{since_hours} hours")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT
                provider,
                COUNT(*) as run_count,
                COALESCE(SUM(total_tasks), 0) as total_tasks,
                COALESCE(SUM(successful_tasks), 0) as successful_tasks,
                COALESCE(SUM(total_tokens), 0) as total_tokens,
                COALESCE(SUM(total_cost_usd), 0) as total_cost,
                COALESCE(AVG(avg_latency_ms), 0) as avg_latency_ms
            FROM runs {where}
            GROUP BY provider
            ORDER BY total_cost DESC
        """
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ── Schedule registry ─────────────────────────────────

    def save_schedule(
        self,
        *,
        schedule_id: str,
        format: str,
        tier: str,
        provider: str,
        model: str | None,
        interval_minutes: int,
        max_iterations: int,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO schedules
                (schedule_id, format, tier, provider, model,
                 interval_minutes, max_iterations, installed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (schedule_id, format, tier, provider, model,
             interval_minutes, max_iterations, now),
        )
        self._conn.commit()

    def remove_schedule(self, schedule_id: str) -> None:
        self._conn.execute(
            "DELETE FROM schedules WHERE schedule_id = ?", (schedule_id,)
        )
        self._conn.commit()

    def get_schedules(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM schedules ORDER BY installed_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_schedule_last_run(
        self, schedule_id: str, status: str
    ) -> None:
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "UPDATE schedules SET last_run_at = ?, last_status = ? "
            "WHERE schedule_id = ?",
            (now, status, schedule_id),
        )
        self._conn.commit()
