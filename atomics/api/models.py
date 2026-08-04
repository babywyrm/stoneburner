"""Pydantic models for the atomics API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# Upper bounds exist so one authenticated request cannot pin the server or run
# up an unbounded provider bill. They are deliberately generous: the CLI is the
# path for genuinely long campaigns and is not affected by these.
MAX_ITERATIONS = 1000
MAX_INTERVAL_SECONDS = 3600
MAX_FIXTURES = 500


class RunRequest(BaseModel):
    """Request body to start a benchmark run."""

    provider: str
    model: str | None = None
    tier: str = "ez"
    iterations: int = Field(default=3, ge=1, le=MAX_ITERATIONS)
    interval: int = Field(default=5, ge=0, le=MAX_INTERVAL_SECONDS)
    save: bool = True


class EvalRequest(BaseModel):
    """Request body to start an eval suite."""

    suite: str
    provider: str
    model: str | None = None
    judge_model: str | None = None
    fixtures: list[str] | None = Field(default=None, max_length=MAX_FIXTURES)
    save: bool = True


class JobResponse(BaseModel):
    """Response representing a job."""

    job_id: str
    status: str
    kind: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error: dict | None = None
    result_url: str | None = None
    result: Any | None = None


class ErrorResponse(BaseModel):
    """Error response."""

    detail: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str


class CompareResponse(BaseModel):
    by: str
    rows: list[dict]


class ReportResponse(BaseModel):
    runs: list[dict]


class ReportSummaryResponse(BaseModel):
    providers: list[dict]
