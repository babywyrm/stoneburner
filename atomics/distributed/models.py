from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


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
    api_key_hint: str | None = None
    status: WorkerStatus = WorkerStatus.ONLINE
    last_seen_at: datetime | None = None
    registered_at: datetime = Field(default_factory=datetime.utcnow)


class WorkerRegisterRequest(BaseModel):
    labels: dict[str, str] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    endpoint: str | None = None


class WorkerRegisterResponse(BaseModel):
    worker_id: str


class DistributedRunRequest(BaseModel):
    mode: JobMode = JobMode.SPLIT
    run_request: dict[str, Any] | None = None
    eval_request: dict[str, Any] | None = None
    worker_selector: dict[str, str] | None = None
    timeout_seconds: int = 600
    max_retries: int = 2


class DistributedJob(BaseModel):
    job_id: str
    mode: JobMode
    parent_run_id: str | None = None
    status: JobStatus = JobStatus.PENDING
    request_json: str
    summary_json: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


class TaskAssignment(BaseModel):
    assignment_id: str
    job_id: str
    worker_id: str | None = None
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
