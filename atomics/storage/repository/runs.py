"""Parent run rows and run listings."""

from __future__ import annotations

from datetime import UTC, datetime

from atomics.models import RunSummary, TaskStatus
from atomics.storage.repository._base import RepositoryBase


class RunsMixin(RepositoryBase):
    def create_run(
        self,
        run_id: str,
        *,
        tier: str = "baseline",
        provider: str = "claude",
        model: str = "",
        trigger: str = "manual",
        pass_count: int = 1,
    ) -> None:
        if pass_count < 1:
            raise ValueError("pass_count must be at least 1")
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "INSERT INTO runs (run_id, started_at, tier, provider, model, trigger, pass_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, now, tier, provider, model, trigger, pass_count),
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

    def get_recent_runs(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def get_run_tasks(self, run_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM task_results WHERE run_id = ? ORDER BY started_at", (run_id,)
        ).fetchall()
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

