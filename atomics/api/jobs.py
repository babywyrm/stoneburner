"""Async job management for the atomics API server."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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
    _task: asyncio.Task[None] | None = field(default=None, repr=False)


class JobManager:
    """In-memory async job manager."""

    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}

    async def submit(
        self,
        kind: str,
        work: Callable[[str], Awaitable[Any]],
    ) -> str:
        job_id = uuid.uuid4().hex
        job = Job(
            job_id=job_id,
            kind=kind,
            status=JobStatus.PENDING,
            created_at=time.time(),
        )
        self.jobs[job_id] = job
        job._task = asyncio.create_task(self._run(job, work))
        return job_id

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
