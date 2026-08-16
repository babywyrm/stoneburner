"""Installed scheduler registry."""

from __future__ import annotations

from datetime import UTC, datetime

from atomics.storage.repository._base import RepositoryBase


class SchedulesMixin(RepositoryBase):
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

