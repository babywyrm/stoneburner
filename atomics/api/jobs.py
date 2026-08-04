"""Async job management for the atomics API server."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from atomics.api.callers import ANONYMOUS_CALLER


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    kind: str
    status: JobStatus
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    result: Any = None
    error: dict[str, Any] | None = None
    # Digest of the API key that submitted this, never the key itself. Used to
    # hold one caller to their share of capacity.
    owner: str = ANONYMOUS_CALLER
    _task: asyncio.Task[None] | None = field(default=None, repr=False)


class TooManyActiveJobsError(RuntimeError):
    """The server is already running its configured maximum of concurrent jobs."""


class CallerQuotaExceededError(TooManyActiveJobsError):
    """One caller is already using their share of concurrent capacity.

    A subclass so routes that catch `TooManyActiveJobsError` keep working: both
    are `429`, and the distinction is in the message the caller reads.
    """


class JobManager:
    """In-memory async job manager.

    Job state lives in this process, so it is bounded in two directions: how
    many jobs may run at once, and how many finished jobs are kept for polling
    afterwards. Without both, a caller who submits in a loop grows the dict
    until the server dies, and every retained result holds a full run summary.

    A third bound is per caller. The global limit alone lets whoever submits
    first take every slot, so a second key gets `429` until the first caller's
    work drains — a denial of service that needs no malice, just one impatient
    script.
    """

    DEFAULT_MAX_ACTIVE = 16
    DEFAULT_MAX_RETAINED = 256
    DEFAULT_MAX_ACTIVE_PER_CALLER = 4

    def __init__(
        self,
        *,
        max_active: int = DEFAULT_MAX_ACTIVE,
        max_retained: int = DEFAULT_MAX_RETAINED,
        max_active_per_caller: int = DEFAULT_MAX_ACTIVE_PER_CALLER,
    ) -> None:
        if max_active < 1:
            raise ValueError(f"max_active must be positive, got {max_active}")
        if max_retained < 1:
            raise ValueError(f"max_retained must be positive, got {max_retained}")
        if max_active_per_caller < 1:
            raise ValueError(
                "max_active_per_caller must be positive, got "
                f"{max_active_per_caller}"
            )
        self.jobs: dict[str, Job] = {}
        self.max_active = max_active
        self.max_retained = max_retained
        self.max_active_per_caller = max_active_per_caller

    @staticmethod
    def _is_active(job: Job) -> bool:
        return job.status in (JobStatus.PENDING, JobStatus.RUNNING)

    @property
    def active_count(self) -> int:
        return sum(1 for job in self.jobs.values() if self._is_active(job))

    def active_count_for(self, owner: str) -> int:
        """Concurrent jobs currently held by one caller."""
        return sum(
            1
            for job in self.jobs.values()
            if job.owner == owner and self._is_active(job)
        )

    async def submit(
        self,
        kind: str,
        work: Callable[[str], Awaitable[Any]],
        *,
        owner: str = ANONYMOUS_CALLER,
    ) -> str:
        if self.active_count >= self.max_active:
            raise TooManyActiveJobsError(
                f"{self.active_count} jobs already running (limit {self.max_active})"
            )
        # Checked after the global limit so a busy server reports the real
        # reason rather than blaming the caller for someone else's load.
        held = self.active_count_for(owner)
        if held >= self.max_active_per_caller:
            raise CallerQuotaExceededError(
                f"caller already has {held} jobs running "
                f"(per-caller limit {self.max_active_per_caller})"
            )
        job_id = uuid.uuid4().hex
        job = Job(
            job_id=job_id,
            kind=kind,
            status=JobStatus.PENDING,
            created_at=time.time(),
            owner=owner,
        )
        self.jobs[job_id] = job
        job._task = asyncio.create_task(self._run(job, work))
        return job_id

    def _evict_finished(self) -> None:
        """Drop the oldest finished jobs once retention is exceeded.

        Only finished jobs are eligible, so a burst of submissions can never
        evict work that is still running.
        """
        finished = [
            job
            for job in self.jobs.values()
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED)
        ]
        excess = len(finished) - self.max_retained
        if excess <= 0:
            return
        finished.sort(key=lambda job: job.completed_at or job.created_at)
        for job in finished[:excess]:
            del self.jobs[job.job_id]

    async def _run(
        self,
        job: Job,
        work: Callable[[str], Awaitable[Any]],
    ) -> None:
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        try:
            job.result = await work(job.job_id)
            job.status = JobStatus.COMPLETED
        except asyncio.CancelledError:
            job.error = {"type": "CancelledError", "message": "job cancelled"}
            job.status = JobStatus.FAILED
            raise
        except Exception as exc:
            job.error = {"type": exc.__class__.__name__, "message": str(exc)}
            job.status = JobStatus.FAILED
        finally:
            job.completed_at = time.time()
            # Pruning on completion rather than on submit keeps the bound exact
            # and still applies once submissions stop.
            self._evict_finished()

    async def wait_for(self, job_id: str, timeout: float | None = None) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job._task is None or job._task.done():
            return job
        if timeout is None:
            await job._task
        elif timeout > 0:
            # asyncio.wait does not cancel the task on timeout (unlike wait_for).
            await asyncio.wait({job._task}, timeout=timeout)
        return job
