from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from atomics.validation import validate_endpoint_url

# Same value as atomics.api.callers.ANONYMOUS_CALLER. Importing it would pull
# FastAPI into every CLI start through atomics.api.
ANONYMOUS_OWNER = "anonymous"
MAX_DISTRIBUTED_ITERATIONS = 1000
MAX_DISTRIBUTED_INTERVAL_SECONDS = 3600
MAX_DISTRIBUTED_TIMEOUT_SECONDS = 86_400
MAX_DISTRIBUTED_RETRIES = 10


class WorkerStatus(StrEnum):
    ONLINE = "online"
    OFFLINE = "offline"
    DISABLED = "disabled"


class AssignmentStatus(StrEnum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PARTIAL = "partial"
    COMPLETED = "completed"
    FAILED = "failed"


class JobMode(StrEnum):
    SPLIT = "split"
    FULL = "full"
    FLEET = "fleet"


class Worker(BaseModel):
    worker_id: str
    labels: dict[str, str] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    endpoint: str | None = None
    status: WorkerStatus = WorkerStatus.ONLINE
    last_seen_at: datetime | None = None
    registered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WorkerRegisterRequest(BaseModel):
    labels: dict[str, str] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    endpoint: str | None = None

    @field_validator("endpoint")
    @classmethod
    def _valid_endpoint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_endpoint_url(value, label="worker endpoint")


class WorkerRegisterResponse(BaseModel):
    worker_id: str


class DistributedRunRequest(BaseModel):
    mode: JobMode = JobMode.SPLIT
    run_request: dict[str, Any] | None = None
    eval_request: dict[str, Any] | None = None
    # Retained so the API can reject it explicitly; dropping the field would
    # make Pydantic ignore it and reintroduce silent no-op targeting.
    worker_selector: dict[str, str] | None = None
    timeout_seconds: int = Field(default=600, ge=1, le=MAX_DISTRIBUTED_TIMEOUT_SECONDS)
    max_retries: int = Field(default=2, ge=0, le=MAX_DISTRIBUTED_RETRIES)

    @field_validator("run_request")
    @classmethod
    def _bounded_run_request(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        if "iterations" in value and value["iterations"] is not None:
            it = value["iterations"]
            if (
                isinstance(it, bool)
                or not isinstance(it, int)
                or not (1 <= it <= MAX_DISTRIBUTED_ITERATIONS)
            ):
                raise ValueError(
                    f"run_request.iterations must be an integer in 1..{MAX_DISTRIBUTED_ITERATIONS}"
                )
        if "interval" in value and value["interval"] is not None:
            iv = value["interval"]
            if (
                isinstance(iv, bool)
                or not isinstance(iv, int)
                or not (0 <= iv <= MAX_DISTRIBUTED_INTERVAL_SECONDS)
            ):
                raise ValueError(
                    "run_request.interval must be an integer in "
                    f"0..{MAX_DISTRIBUTED_INTERVAL_SECONDS}"
                )
        return value


class DistributedJob(BaseModel):
    job_id: str
    mode: JobMode
    parent_run_id: str | None = None
    status: JobStatus = JobStatus.PENDING
    request_json: str
    summary_json: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    # Caller digest that submitted the job. Scopes reads; never serialized.
    owner: str = Field(default=ANONYMOUS_OWNER, exclude=True)


class TaskAssignment(BaseModel):
    assignment_id: str
    job_id: str
    worker_id: str | None = None
    # None: claimable by any worker (split). Set: only this worker may run it
    # (fleet), and it is never reassigned elsewhere.
    target_worker_id: str | None = None
    status: AssignmentStatus = AssignmentStatus.PENDING
    task_spec: dict[str, Any]
    result_json: str | None = None
    retry_count: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None


class TaskResultSubmission(BaseModel):
    status: AssignmentStatus
    result_json: str | None = None
    error: str | None = None
