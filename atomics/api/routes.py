"""FastAPI routes for the atomics API server."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from atomics.api._runners import (
    run_benchmark_from_request,
    run_eval_suite,
    validate_eval_suite,
)
from atomics.api.auth import AuthBackend
from atomics.api.jobs import Job, JobManager
from atomics.api.models import (
    CompareResponse,
    EvalRequest,
    HealthResponse,
    JobResponse,
    ReportResponse,
    RunRequest,
)
from atomics.config import load_settings
from atomics.storage.repository import MetricsRepository

router = APIRouter(prefix="/api/v1")


def get_job_manager(request: Request) -> JobManager:
    return request.app.state.job_manager


def get_auth(request: Request) -> AuthBackend:
    return request.app.state.auth


async def require_auth(request: Request, auth: AuthBackend = Depends(get_auth)) -> None:
    if not await auth.authenticate(request):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/runs", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_run(
    payload: RunRequest,
    job_manager: JobManager = Depends(get_job_manager),
    _: None = Depends(require_auth),
) -> JobResponse:
    job_id = await job_manager.submit(
        "run", lambda _jid: run_benchmark_from_request(payload)
    )
    job = job_manager.jobs[job_id]
    return _job_to_response(job)


@router.post("/evals", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_eval(
    payload: EvalRequest,
    job_manager: JobManager = Depends(get_job_manager),
    _: None = Depends(require_auth),
) -> JobResponse:
    try:
        validate_eval_suite(payload.suite)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    job_id = await job_manager.submit(
        "eval", lambda _jid: run_eval_suite(payload)
    )
    job = job_manager.jobs[job_id]
    return _job_to_response(job)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
    _: None = Depends(require_auth),
) -> JobResponse:
    job = job_manager.jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    return _job_to_response(job)


@router.get("/compare", response_model=CompareResponse)
async def compare(
    by: str = "provider",
    since_hours: float | None = None,
    tier: str | None = None,
    category: str | None = None,
    _: None = Depends(require_auth),
) -> CompareResponse:
    if by not in {"provider", "model"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="by must be 'provider' or 'model'",
        )
    settings = load_settings()
    repo = MetricsRepository(settings.db_path)
    try:
        rows = repo.compare_providers(
            since_hours=since_hours,
            tier=tier,
            category=category,
            group_by=by,
        )
        return CompareResponse(by=by, rows=rows)
    finally:
        repo.close()


@router.get("/reports/recent-runs", response_model=ReportResponse)
async def recent_runs(
    limit: int = 10,
    _: None = Depends(require_auth),
) -> ReportResponse:
    settings = load_settings()
    repo = MetricsRepository(settings.db_path)
    try:
        rows = repo.get_recent_runs(limit=limit)
        return ReportResponse(runs=rows)
    finally:
        repo.close()


def _job_to_response(job: Job) -> JobResponse:
    return JobResponse(
        job_id=job.job_id,
        status=job.status.value,
        kind=job.kind,
        created_at=str(job.created_at),
        started_at=str(job.started_at) if job.started_at is not None else None,
        completed_at=str(job.completed_at) if job.completed_at is not None else None,
        error=job.error,
        result=job.result,
    )
