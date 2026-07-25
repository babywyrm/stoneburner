"""Coordinator state machine for distributed runs."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from sqlite3 import Connection
from typing import Any

from atomics.distributed.models import (
    AssignmentStatus,
    DistributedJob,
    DistributedRunRequest,
    JobMode,
    JobStatus,
    TaskAssignment,
    Worker,
    WorkerRegisterRequest,
    WorkerStatus,
)


class Coordinator:
    """Manage workers, jobs, and task assignments in SQLite."""

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()

    def register_worker(
        self, req: WorkerRegisterRequest, *, api_key_hint: str | None = None
    ) -> Worker:
        worker_id = uuid.uuid4().hex[:12]
        now = self._now()
        now_dt = datetime.now(UTC)
        worker = Worker(
            worker_id=worker_id,
            labels=req.labels,
            capabilities=req.capabilities,
            endpoint=req.endpoint,
            api_key_hint=api_key_hint,
            status=WorkerStatus.ONLINE,
            last_seen_at=now_dt,
            registered_at=now_dt,
        )
        self._conn.execute(
            "INSERT INTO workers "
            "(worker_id, labels, capabilities, endpoint, api_key_hint, "
            "status, last_seen_at, registered_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                worker.worker_id,
                json.dumps(worker.labels),
                json.dumps(worker.capabilities),
                worker.endpoint,
                worker.api_key_hint,
                worker.status.value,
                now,
                now,
            ),
        )
        self._conn.commit()
        return worker

    def heartbeat(self, worker_id: str) -> Worker | None:
        now = self._now()
        self._conn.execute(
            "UPDATE workers SET status = ?, last_seen_at = ? WHERE worker_id = ?",
            (WorkerStatus.ONLINE.value, now, worker_id),
        )
        self._conn.commit()
        return self.get_worker(worker_id)

    WORKER_COLUMNS = (
        "worker_id, labels, capabilities, endpoint, api_key_hint, status, "
        "last_seen_at, registered_at"
    )

    def get_worker(self, worker_id: str) -> Worker | None:
        row = self._conn.execute(
            f"SELECT {self.WORKER_COLUMNS} FROM workers WHERE worker_id = ?",
            (worker_id,),
        ).fetchone()
        if not row:
            return None
        return self._row_to_worker(row)

    def matching_workers(self, selector: dict[str, str] | None) -> list[Worker]:
        """Online workers whose labels satisfy every pair in the selector.

        An empty or absent selector matches every online worker. Matching is exact
        key/value equality, and extra labels on a worker do not disqualify it.

        Offline workers are excluded: a fleet run broadcasts to the hosts that are
        actually present when it is submitted, and pinning work to an absent worker
        would produce assignments nothing can ever claim. Liveness is refreshed
        first so a host that stopped heartbeating is not handed a fresh slice.
        """
        self._mark_absent_workers()
        rows = self._conn.execute(
            f"SELECT {self.WORKER_COLUMNS} FROM workers WHERE status = ? "
            "ORDER BY registered_at, worker_id",
            (WorkerStatus.ONLINE.value,),
        ).fetchall()
        workers = [self._row_to_worker(row) for row in rows]
        if not selector:
            return workers
        return [
            worker
            for worker in workers
            if all(worker.labels.get(key) == value for key, value in selector.items())
        ]

    def _row_to_worker(self, row: Any) -> Worker:
        return Worker(
            worker_id=row[0],
            labels=json.loads(row[1]),
            capabilities=json.loads(row[2]) if row[2] else [],
            endpoint=row[3],
            api_key_hint=row[4],
            status=WorkerStatus(row[5]),
            last_seen_at=datetime.fromisoformat(row[6]) if row[6] else None,
            registered_at=datetime.fromisoformat(row[7]),
        )

    def _insert_job(
        self, request: DistributedRunRequest, mode: JobMode
    ) -> DistributedJob:
        """Insert the job row. Caller adds assignments, then commits."""
        parent_run_id = None
        if request.run_request:
            run_id = request.run_request.get("run_id")
            if isinstance(run_id, str) and run_id:
                parent_run_id = run_id
        if not parent_run_id:
            parent_run_id = uuid.uuid4().hex[:12]
        job = DistributedJob(
            job_id=uuid.uuid4().hex[:12],
            mode=mode,
            parent_run_id=parent_run_id,
            status=JobStatus.PENDING,
            request_json=request.model_dump_json(),
            created_at=datetime.now(UTC),
        )
        self._conn.execute(
            "INSERT INTO distributed_jobs "
            "(job_id, mode, parent_run_id, status, request_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                job.job_id,
                job.mode.value,
                job.parent_run_id,
                job.status.value,
                job.request_json,
                self._now(),
            ),
        )
        return job

    def _insert_assignment(
        self,
        job_id: str,
        spec: dict[str, Any],
        *,
        target_worker_id: str | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO distributed_assignments "
            "(assignment_id, job_id, status, task_spec, target_worker_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                uuid.uuid4().hex[:12],
                job_id,
                AssignmentStatus.PENDING.value,
                json.dumps(spec),
                target_worker_id,
            ),
        )

    def create_split_job(
        self, request: DistributedRunRequest, task_specs: list[dict[str, Any]]
    ) -> DistributedJob:
        """One assignment per task, claimable by whichever worker asks first."""
        job = self._insert_job(request, JobMode.SPLIT)
        for spec in task_specs:
            self._insert_assignment(job.job_id, spec)
        self._conn.commit()
        return job

    def create_fleet_job(
        self,
        request: DistributedRunRequest,
        task_specs: list[dict[str, Any]],
        workers: list[Worker],
    ) -> DistributedJob:
        """Broadcast one task set to every worker, pinned per host.

        `task_specs` is reused across workers rather than regenerated, which is
        what makes the hosts comparable: the caller must build it once. Producing
        specs per worker would yield the right assignment count while quietly
        giving each host different prompts.
        """
        job = self._insert_job(request, JobMode.FLEET)
        for worker in workers:
            for spec in task_specs:
                self._insert_assignment(
                    job.job_id, spec, target_worker_id=worker.worker_id
                )
        self._conn.commit()
        return job

    def claim_assignment(self, worker_id: str) -> TaskAssignment | None:
        self._requeue_stale_assignments()
        cursor = self._conn.execute(
            """
            UPDATE distributed_assignments
            SET status = ?, worker_id = ?, started_at = ?,
                retry_count = retry_count + 1
            WHERE assignment_id = (
                SELECT assignment_id FROM distributed_assignments
                WHERE status = ?
                  AND (target_worker_id IS NULL OR target_worker_id = ?)
                ORDER BY assignment_id
                LIMIT 1
            )
            RETURNING assignment_id, job_id, worker_id, target_worker_id, status,
                      task_spec, result_json, retry_count, started_at,
                      completed_at
            """,
            (
                AssignmentStatus.ASSIGNED.value,
                worker_id,
                self._now(),
                AssignmentStatus.PENDING.value,
                worker_id,
            ),
        )
        row = cursor.fetchone()
        if not row:
            self._conn.commit()
            return None
        self._conn.execute(
            "UPDATE distributed_jobs SET status = ? "
            "WHERE job_id = ? AND status = ?",
            (JobStatus.RUNNING.value, row[1], JobStatus.PENDING.value),
        )
        self._conn.commit()
        return self._row_to_assignment(row)

    ASSIGNMENT_COLUMNS = (
        "assignment_id, job_id, worker_id, target_worker_id, status, task_spec, "
        "result_json, retry_count, started_at, completed_at"
    )

    def _row_to_assignment(self, row: Any) -> TaskAssignment:
        return TaskAssignment(
            assignment_id=row[0],
            job_id=row[1],
            worker_id=row[2],
            target_worker_id=row[3],
            status=AssignmentStatus(row[4]),
            task_spec=json.loads(row[5]),
            result_json=row[6],
            retry_count=row[7],
            started_at=datetime.fromisoformat(row[8]) if row[8] else None,
            completed_at=datetime.fromisoformat(row[9]) if row[9] else None,
        )

    def submit_assignment(
        self,
        assignment_id: str,
        result_json: str | None,
        *,
        error: str | None = None,
    ) -> TaskAssignment | None:
        status = (
            AssignmentStatus.FAILED.value
            if error
            else AssignmentStatus.COMPLETED.value
        )
        self._conn.execute(
            "UPDATE distributed_assignments "
            "SET status = ?, result_json = ?, completed_at = ? "
            "WHERE assignment_id = ?",
            (status, result_json, self._now(), assignment_id),
        )
        self._conn.commit()
        assignment = self.get_assignment(assignment_id)
        if assignment:
            self._update_job_status(assignment.job_id)
        return assignment

    def get_assignment(self, assignment_id: str) -> TaskAssignment | None:
        row = self._conn.execute(
            f"SELECT {self.ASSIGNMENT_COLUMNS} "
            "FROM distributed_assignments WHERE assignment_id = ?",
            (assignment_id,),
        ).fetchone()
        if not row:
            return None
        return self._row_to_assignment(row)

    def get_job(self, job_id: str) -> DistributedJob | None:
        row = self._conn.execute(
            "SELECT job_id, mode, parent_run_id, status, request_json, "
            "summary_json, created_at, completed_at "
            "FROM distributed_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if not row:
            return None
        return DistributedJob(
            job_id=row[0],
            mode=JobMode(row[1]),
            parent_run_id=row[2],
            status=JobStatus(row[3]),
            request_json=row[4],
            summary_json=row[5],
            created_at=datetime.fromisoformat(row[6]),
            completed_at=datetime.fromisoformat(row[7]) if row[7] else None,
        )

    def _update_job_status(self, job_id: str) -> None:
        rows = self._conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) as failed
            FROM distributed_assignments WHERE job_id = ?
            """,
            (
                AssignmentStatus.COMPLETED.value,
                AssignmentStatus.FAILED.value,
                job_id,
            ),
        ).fetchone()
        total, completed, failed = rows[0], rows[1] or 0, rows[2] or 0
        if total > 0 and completed + failed == total:
            new_status = (
                JobStatus.COMPLETED.value
                if failed == 0
                else JobStatus.PARTIAL.value
            )
            self._conn.execute(
                "UPDATE distributed_jobs "
                "SET status = ?, completed_at = ? WHERE job_id = ?",
                (new_status, self._now(), job_id),
            )
            self._conn.commit()

    def _timeout_seconds_for_job(self, request_json: str) -> int:
        try:
            payload = json.loads(request_json)
        except json.JSONDecodeError:
            return 600
        timeout = payload.get("timeout_seconds", 600)
        if isinstance(timeout, int) and timeout > 0:
            return timeout
        return 600

    # Roughly four missed heartbeats at the worker's default 30s interval. Chosen
    # to be forgiving: the cost of declaring a live worker absent is failing work
    # it could have finished, while the cost of waiting is only a slower verdict.
    WORKER_ABSENT_AFTER_SECONDS = 120

    def _mark_absent_workers(self) -> None:
        """Mark online workers that have stopped heartbeating as offline.

        Nothing else in the coordinator ever set OFFLINE, so a killed worker stayed
        'online' indefinitely. Fleet mode cannot tolerate that: its assignments are
        claimable only by their target, so an absent target means a job waiting on
        a host that is never coming back.
        """
        cutoff = datetime.now(UTC) - timedelta(seconds=self.WORKER_ABSENT_AFTER_SECONDS)
        self._conn.execute(
            "UPDATE workers SET status = ? "
            "WHERE status = ? AND last_seen_at IS NOT NULL AND last_seen_at < ?",
            (
                WorkerStatus.OFFLINE.value,
                WorkerStatus.ONLINE.value,
                cutoff.isoformat(),
            ),
        )
        self._conn.commit()

    def _fail_absent_worker_slices(self) -> set[str]:
        """Fail pinned work belonging to offline workers. Returns affected job ids.

        Handles the host that died before claiming anything: those assignments are
        still `pending`, so the stale-assignment scan never looks at them, and the
        job would wait forever on a worker that cannot ask for work.
        """
        rows = self._conn.execute(
            "SELECT DISTINCT a.job_id, a.target_worker_id "
            "FROM distributed_assignments a "
            "JOIN workers w ON w.worker_id = a.target_worker_id "
            "WHERE a.status IN (?, ?) AND w.status = ?",
            (
                AssignmentStatus.PENDING.value,
                AssignmentStatus.ASSIGNED.value,
                WorkerStatus.OFFLINE.value,
            ),
        ).fetchall()
        for job_id, worker_id in rows:
            self._fail_pinned_slice(job_id, worker_id)
        return {job_id for job_id, _ in rows}

    def _max_retries_for_job(self, request_json: str) -> int:
        try:
            payload = json.loads(request_json)
        except json.JSONDecodeError:
            return 2
        retries = payload.get("max_retries", 2)
        if isinstance(retries, int) and retries >= 0:
            return retries
        return 2

    def _fail_pinned_slice(self, job_id: str, worker_id: str) -> None:
        """Fail every unfinished assignment of this job pinned to one worker.

        Pinned work is claimable only by its target, so once that host is judged
        unable to finish, leaving the rest pending would hang the job behind a
        worker that is never coming back. Failing the slice as a unit lets the job
        resolve to PARTIAL with the host's losses visible, while its tasks are
        still never executed anywhere else.
        """
        self._conn.execute(
            "UPDATE distributed_assignments SET status = ?, completed_at = ? "
            "WHERE job_id = ? AND target_worker_id = ? AND status IN (?, ?)",
            (
                AssignmentStatus.FAILED.value,
                self._now(),
                job_id,
                worker_id,
                AssignmentStatus.PENDING.value,
                AssignmentStatus.ASSIGNED.value,
            ),
        )

    def _requeue_stale_assignments(self) -> None:
        self._mark_absent_workers()
        failed_jobs = self._fail_absent_worker_slices()
        rows = self._conn.execute(
            """
            SELECT a.assignment_id, a.started_at, a.target_worker_id, a.job_id,
                   a.retry_count, j.request_json, w.status
            FROM distributed_assignments a
            JOIN distributed_jobs j ON j.job_id = a.job_id
            LEFT JOIN workers w ON w.worker_id = a.worker_id
            WHERE a.status = ?
            """,
            (AssignmentStatus.ASSIGNED.value,),
        ).fetchall()
        now = datetime.now(UTC)
        for (
            assignment_id,
            started_at,
            target_worker_id,
            job_id,
            retry_count,
            request_json,
            worker_status,
        ) in rows:
            worker_gone = (
                worker_status is None or worker_status == WorkerStatus.OFFLINE.value
            )
            timed_out = False
            if not worker_gone and started_at:
                started = datetime.fromisoformat(started_at)
                if started.tzinfo is None:
                    started = started.replace(tzinfo=UTC)
                timeout = self._timeout_seconds_for_job(request_json)
                timed_out = (now - started).total_seconds() > timeout
            if not (worker_gone or timed_out):
                continue

            if target_worker_id:
                if worker_gone:
                    # The host is definitively absent, so its whole slice dies
                    # with it rather than waiting on a worker that has left.
                    self._fail_pinned_slice(job_id, target_worker_id)
                    failed_jobs.add(job_id)
                    continue
                if retry_count > self._max_retries_for_job(request_json):
                    # Out of attempts on this task. Only this assignment fails:
                    # the host is still heartbeating, so the rest of its slice
                    # may well succeed.
                    self._conn.execute(
                        "UPDATE distributed_assignments "
                        "SET status = ?, completed_at = ? WHERE assignment_id = ?",
                        (AssignmentStatus.FAILED.value, self._now(), assignment_id),
                    )
                    failed_jobs.add(job_id)
                    continue

            self._conn.execute(
                """
                UPDATE distributed_assignments
                SET status = ?, worker_id = NULL, started_at = NULL
                WHERE assignment_id = ?
                """,
                (AssignmentStatus.PENDING.value, assignment_id),
            )
        self._conn.commit()
        # After failing work, the job may now be terminal.
        for job_id in failed_jobs:
            self._update_job_status(job_id)

    def recover_jobs(self) -> None:
        """Re-queue assigned work for offline workers or stale started_at."""
        self._requeue_stale_assignments()
