"""Pydantic models for the atomics API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    """Request body to start a benchmark run."""

    provider: str
    model: str | None = None
    tier: str = "ez"
    iterations: int = Field(default=3, ge=1)
    interval: int = Field(default=5, ge=0)
    save: bool = True


class EvalRequest(BaseModel):
    """Request body to start an eval suite."""

    suite: str
    provider: str
    model: str | None = None
    judge_model: str | None = None
    fixtures: list[str] | None = None
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
