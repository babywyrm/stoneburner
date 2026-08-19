"""FastAPI routes for the atomics API server."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from atomics.api._discovery import list_models, run_provider_test
from atomics.api._runners import (
    run_benchmark_from_request,
    run_eval_suite,
    validate_eval_suite,
)
from atomics.api._sweep import run_sweep_from_request
from atomics.api.dependencies import require_auth
from atomics.api.jobs import Job, JobManager, TooManyActiveJobsError
from atomics.api.models import (
    MAX_TREND_HOURS,
    CompareResponse,
    EvalRequest,
    HealthResponse,
    JobResponse,
    JobsListResponse,
    JobSummary,
    ModelsResponse,
    ProviderTestRequest,
    ProviderTestResponse,
    ReadinessCheck,
    ReadinessResponse,
    ReportResponse,
    RunRequest,
    SweepRequest,
    TrendsResponse,
)
from atomics.config import load_settings
from atomics.storage.repository import MetricsRepository

router = APIRouter(prefix="/api/v1")


def get_job_manager(request: Request) -> JobManager:
    return request.app.state.job_manager


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness. Answers `ok` whenever this process can serve a request.

    Intentionally free of dependency checks. Wiring the database into liveness
    would have an orchestrator restart the API server during a database
    outage — which repairs nothing and destroys the readiness signal and the
    in-flight jobs along with it. Point liveness probes here and readiness
    probes at `/ready`.
    """
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadinessResponse)
async def ready(request: Request, response: Response) -> ReadinessResponse:
    """Readiness. Reports `503` when a dependency this server needs is down.

    Split out from `/health`, which used to answer `ok` unconditionally and so
    kept a server in a load balancer's rotation while every request it received
    was going to fail on an unreachable coordinator database.
    """
    checks = [_database_check(request)]
    ready_now = all(check.ok for check in checks)
    if not ready_now:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ready" if ready_now else "not_ready",
        checks=checks,
    )


def _database_check(request: Request) -> ReadinessCheck:
    coordinator = getattr(request.app.state, "coordinator", None)
    if coordinator is None:
        # Reachable before startup completes, so report not-ready rather than
        # raising: an unready server is exactly what this endpoint describes.
        return ReadinessCheck(
            name="database", ok=False, detail="coordinator not initialized"
        )
    error = coordinator.check_database()
    return ReadinessCheck(name="database", ok=error is None, detail=error)


@router.post("/runs", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_run(
    payload: RunRequest,
    job_manager: JobManager = Depends(get_job_manager),
    caller: str = Depends(require_auth),
) -> JobResponse:
    try:
        job_id = await job_manager.submit(
            "run", lambda _jid: run_benchmark_from_request(payload), owner=caller
        )
    except TooManyActiveJobsError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)
        ) from exc
    job = job_manager.jobs[job_id]
    return _job_to_response(job)


@router.post("/evals", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_eval(
    payload: EvalRequest,
    job_manager: JobManager = Depends(get_job_manager),
    caller: str = Depends(require_auth),
) -> JobResponse:
    try:
        validate_eval_suite(payload.suite)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    try:
        job_id = await job_manager.submit(
            "eval", lambda _jid: run_eval_suite(payload), owner=caller
        )
    except TooManyActiveJobsError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)
        ) from exc
    job = job_manager.jobs[job_id]
    return _job_to_response(job)


@router.post("/sweeps", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_sweep(
    payload: SweepRequest,
    job_manager: JobManager = Depends(get_job_manager),
    caller: str = Depends(require_auth),
) -> JobResponse:
    """Start a bounded multi-model, multi-suite campaign. Budget is required."""
    try:
        job_id = await job_manager.submit(
            "sweep", lambda _jid: run_sweep_from_request(payload), owner=caller
        )
    except TooManyActiveJobsError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)
        ) from exc
    job = job_manager.jobs[job_id]
    return _job_to_response(job)


@router.get("/jobs", response_model=JobsListResponse)
async def list_jobs(
    job_manager: JobManager = Depends(get_job_manager),
    _: None = Depends(require_auth),
) -> JobsListResponse:
    """In-memory API jobs. The result payload is omitted; poll `/jobs/{id}`."""
    return JobsListResponse(
        jobs=[_job_to_summary(job) for job in job_manager.list_jobs()]
    )


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


@router.get("/models", response_model=ModelsResponse)
async def get_models(
    provider: str = "ollama",
    host: str | None = None,
    _: None = Depends(require_auth),
) -> ModelsResponse:
    """List tags on an Ollama or vLLM instance. Does not generate."""
    body = await list_models(provider, host)
    return ModelsResponse(**body)


@router.post("/provider-test", response_model=ProviderTestResponse)
async def provider_test(
    payload: ProviderTestRequest,
    _: None = Depends(require_auth),
) -> ProviderTestResponse:
    """Health-check a provider and generate a fixed 2+2 probe. Spends tokens."""
    body = await run_provider_test(payload)
    return ProviderTestResponse(**body)


@router.get("/runs/{run_id}")
async def get_run(
    run_id: str,
    request: Request,
    _: None = Depends(require_auth),
) -> dict:
    """One persisted run and its fixtures. Prompts and raw JSON are omitted."""
    settings = request.app.state.settings
    repo = MetricsRepository(settings.db_path)
    try:
        detail = repo.get_run_detail(run_id)
    finally:
        repo.close()
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found",
        )
    return detail


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


@router.get("/reports/trends", response_model=TrendsResponse)
async def trends(
    request: Request,
    hours: int = Query(default=24, ge=1, le=MAX_TREND_HOURS),
    _: None = Depends(require_auth),
) -> TrendsResponse:
    """Hourly token and cost series. Prompts are not included."""
    settings = request.app.state.settings
    repo = MetricsRepository(settings.db_path)
    try:
        rows = repo.get_token_usage_by_hour(hours=hours)
        return TrendsResponse(hours=hours, rows=rows)
    finally:
        repo.close()


def _job_to_summary(job: Job) -> JobSummary:
    return JobSummary(
        job_id=job.job_id,
        status=job.status.value,
        kind=job.kind,
        created_at=str(job.created_at),
        started_at=str(job.started_at) if job.started_at is not None else None,
        completed_at=str(job.completed_at) if job.completed_at is not None else None,
        error=job.error,
    )


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
