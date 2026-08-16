"""Pydantic models for the atomics API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Upper bounds exist so one authenticated request cannot pin the server or run
# up an unbounded provider bill. They are deliberately generous: the CLI is the
# path for genuinely long campaigns and is not affected by these.
MAX_ITERATIONS = 1000
MAX_INTERVAL_SECONDS = 3600
MAX_FIXTURES = 500
MAX_TREND_HOURS = 168

# Dollar ceiling for one API-triggered eval, shared by the model and its judge.
# Eval suites are the only remotely reachable path that spends without a tier
# profile behind it, so unlike the CLI this is always applied.
DEFAULT_EVAL_BUDGET_USD = 10.0
MAX_EVAL_BUDGET_USD = 1000.0


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
    budget_usd: float = Field(
        default=DEFAULT_EVAL_BUDGET_USD, gt=0, le=MAX_EVAL_BUDGET_USD
    )


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
    """Liveness response: this process is running and serving requests.

    Deliberately checks nothing external. A liveness probe answers "should this
    process be restarted", and restarting the API server does not repair an
    unreachable database — it just removes the endpoint that could have told
    you what was wrong. Dependency checks belong on `/ready`.
    """

    status: str


class ReadinessCheck(BaseModel):
    """One dependency's contribution to readiness."""

    name: str
    ok: bool
    detail: str | None = None


class ReadinessResponse(BaseModel):
    """Readiness: whether this server can currently serve real work."""

    status: str
    checks: list[ReadinessCheck]


class CompareResponse(BaseModel):
    by: str
    rows: list[dict]


class ReportResponse(BaseModel):
    runs: list[dict]


class TrendsResponse(BaseModel):
    hours: int
    rows: list[dict]


class JobSummary(BaseModel):
    """In-memory job without the result payload."""

    job_id: str
    status: str
    kind: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error: dict | None = None


class JobsListResponse(BaseModel):
    jobs: list[JobSummary]


class ReportSummaryResponse(BaseModel):
    providers: list[dict]


class ModelsResponse(BaseModel):
    provider: str
    models: list[dict]


class ProviderTestRequest(BaseModel):
    """Probe a provider. The generate prompt is fixed server-side."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str | None = None
    host: str | None = None
    thinking: bool | None = None


class ProviderTestResponse(BaseModel):
    ok: bool
    health: bool
    provider: str
    model: str | None = None
    response: str | None = None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    thinking_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
