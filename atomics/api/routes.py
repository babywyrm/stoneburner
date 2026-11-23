"""FastAPI routes for the atomics API server."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from atomics.api._battery import run_battery_from_request
from atomics.api._discovery import list_models, run_provider_test
from atomics.api._load import run_soak_from_request, run_stress_from_request
from atomics.api._runners import (
    run_benchmark_from_request,
    run_eval_suite,
    validate_eval_suite,
)
from atomics.api._sweep import run_sweep_from_request
from atomics.api.config import DisallowedHostError
from atomics.api.dependencies import require_auth
from atomics.api.job_progress import (
    initial_battery_progress,
    initial_eval_progress,
    initial_run_progress,
    initial_soak_progress,
    initial_stress_progress,
    initial_sweep_progress,
    payload_request,
    resolve_eval_request,
    short_request,
)
from atomics.api.jobs import Job, JobManager, TooManyActiveJobsError
from atomics.api.models import (
    MAX_TREND_HOURS,
    BatteryRequest,
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
    SoakRequest,
    StressRequest,
    SweepRequest,
    TrendsResponse,
)
from atomics.config import load_settings
from atomics.storage.repository import MetricsRepository

router = APIRouter(prefix="/api/v1")


def get_job_manager(request: Request) -> JobManager:
    from typing import cast

    return cast(JobManager, request.app.state.job_manager)


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
        return ReadinessCheck(name="database", ok=False, detail="coordinator not initialized")
    error = coordinator.check_database()
    return ReadinessCheck(name="database", ok=error is None, detail=error)


def _check_payload_hosts(request: Request, *pairs: tuple[str, str | None]) -> None:
    settings = request.app.state.settings
    try:
        for label, url in pairs:
            settings.check_outbound_host(url, label=label)
    except DisallowedHostError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


def _check_provider_hosts(
    request: Request,
    provider: str,
    host: str | None,
    *,
    include_judge: bool = False,
    judge_host: str | None = None,
) -> None:
    """Enforce `allowed_hosts` against both explicit and default/vendor endpoints."""
    _check_payload_hosts(request, ("host", host), ("judge_host", judge_host))
    if request.app.state.settings.allowed_hosts is None:
        return
    from atomics.providers.factory import ProviderConfigError, resolve_provider_endpoint

    app_settings = load_settings()
    try:
        eff_host = resolve_provider_endpoint(provider, host, app_settings, host_label="host")
    except ProviderConfigError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    _check_payload_hosts(request, ("host", eff_host))

    if include_judge:
        eff_judge_in = (
            judge_host if judge_host is not None else (host if provider == "ollama" else None)
        )
        try:
            eff_judge = resolve_provider_endpoint(
                "ollama", eff_judge_in, app_settings, host_label="judge_host"
            )
        except ProviderConfigError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        _check_payload_hosts(request, ("judge_host", eff_judge))


@router.post("/runs", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_run(
    request: Request,
    payload: RunRequest,
    job_manager: JobManager = Depends(get_job_manager),
    caller: str = Depends(require_auth),
) -> JobResponse:
    import inspect

    _check_provider_hosts(request, payload.provider, payload.host)
    db_path = request.app.state.settings.db_path
    accepts_db = "db_path" in inspect.signature(run_benchmark_from_request).parameters

    async def _work(jid: str) -> dict[str, Any]:
        job_obj = job_manager.jobs[jid]
        if accepts_db:
            return await run_benchmark_from_request(payload, job=job_obj, db_path=db_path)
        return await run_benchmark_from_request(payload, job=job_obj)

    try:
        job_id = await job_manager.submit(
            "run",
            _work,
            owner=caller,
            request=payload_request(payload, load_settings()),
            progress=initial_run_progress(payload),
        )
    except TooManyActiveJobsError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    job = job_manager.jobs[job_id]
    return _job_to_response(job)


@router.post("/evals", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_eval(
    request: Request,
    payload: EvalRequest,
    job_manager: JobManager = Depends(get_job_manager),
    caller: str = Depends(require_auth),
) -> JobResponse:
    import inspect

    try:
        suite = validate_eval_suite(payload.suite)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    if suite == "codegen" and not request.app.state.settings.allow_codegen:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "codegen suite is disabled on this server; start with "
                "--allow-codegen to enable remote code execution"
            ),
        )
    _check_provider_hosts(
        request,
        payload.provider,
        payload.host,
        include_judge=suite != "codegen",
        judge_host=payload.judge_host,
    )
    db_path = request.app.state.settings.db_path
    accepts_db = "db_path" in inspect.signature(run_eval_suite).parameters

    async def _work(jid: str) -> dict[str, Any]:
        job_obj = job_manager.jobs[jid]
        if accepts_db:
            return await run_eval_suite(payload, job=job_obj, db_path=db_path)
        return await run_eval_suite(payload, job=job_obj)

    try:
        job_id = await job_manager.submit(
            "eval",
            _work,
            owner=caller,
            request=resolve_eval_request(payload, load_settings()),
            progress=initial_eval_progress(payload),
        )
    except TooManyActiveJobsError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    job = job_manager.jobs[job_id]
    return _job_to_response(job)


@router.post("/sweeps", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_sweep(
    request: Request,
    payload: SweepRequest,
    job_manager: JobManager = Depends(get_job_manager),
    caller: str = Depends(require_auth),
) -> JobResponse:
    """Start a bounded multi-model, multi-suite campaign. Budget is required."""
    import inspect

    _check_provider_hosts(
        request,
        payload.provider,
        payload.host,
        include_judge=True,
        judge_host=payload.judge_host,
    )
    db_path = request.app.state.settings.db_path
    accepts_db = "db_path" in inspect.signature(run_sweep_from_request).parameters

    async def _work(jid: str) -> dict[str, Any]:
        job_obj = job_manager.jobs[jid]
        if accepts_db:
            return await run_sweep_from_request(payload, job=job_obj, db_path=db_path)
        return await run_sweep_from_request(payload, job=job_obj)

    try:
        job_id = await job_manager.submit(
            "sweep",
            _work,
            owner=caller,
            request=payload_request(payload, load_settings()),
            progress=initial_sweep_progress(payload),
        )
    except TooManyActiveJobsError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    job = job_manager.jobs[job_id]
    return _job_to_response(job)


@router.post("/batteries", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_battery(
    request: Request,
    payload: BatteryRequest,
    job_manager: JobManager = Depends(get_job_manager),
    caller: str = Depends(require_auth),
) -> JobResponse:
    """Run a named battery as one metered job. Budget is required."""
    import inspect

    from atomics.benchmark.qa_runner import load_qa_suite
    from atomics.eval.batteries import get_battery, visible_steps

    needs_judge = payload.judge_host is not None
    qa_hosts: list[str] = []
    if request.app.state.settings.allowed_hosts is not None:
        try:
            bat = get_battery(payload.name)
            steps = (
                *visible_steps(bat, provider=payload.provider, profile=payload.profile),
                *bat.optional_steps,
            )
        except KeyError:
            steps = ()
        needs_judge = needs_judge or any(
            s.suite not in {"provider-test", "qa", "archreview", "codegen"} for s in steps
        )
        if payload.profile is None:
            for s in steps:
                if s.suite == "qa" and s.qa_file:
                    _fm, file_host, _ = load_qa_suite(s.qa_file)
                    if payload.host or file_host:
                        qa_hosts.append(payload.host or file_host)

    _check_provider_hosts(
        request,
        payload.provider,
        payload.host,
        include_judge=needs_judge,
        judge_host=payload.judge_host,
    )
    for qh in qa_hosts:
        _check_payload_hosts(request, ("qa host", qh))
    if payload.profile is not None:
        from atomics.load.profiles import load_confined_profile

        try:
            prof = load_confined_profile(payload.profile)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        prof_url = prof.ollama_host if prof.type == "ollama" else prof.http_url
        _check_payload_hosts(request, ("profile target", prof_url))

    db_path = request.app.state.settings.db_path
    accepts_db = "db_path" in inspect.signature(run_battery_from_request).parameters

    async def _work(jid: str) -> dict[str, Any]:
        job_obj = job_manager.jobs[jid]
        if accepts_db:
            return await run_battery_from_request(payload, job=job_obj, db_path=db_path)
        return await run_battery_from_request(payload, job=job_obj)

    try:
        job_id = await job_manager.submit(
            "battery",
            _work,
            owner=caller,
            request=payload_request(payload, load_settings()),
            progress=initial_battery_progress(payload),
        )
    except TooManyActiveJobsError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    job = job_manager.jobs[job_id]
    return _job_to_response(job)


@router.post("/stress", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_stress(
    request: Request,
    payload: StressRequest,
    job_manager: JobManager = Depends(get_job_manager),
    caller: str = Depends(require_auth),
) -> JobResponse:
    """Ramp concurrency to find saturation. One model, required budget, capped."""
    _check_provider_hosts(request, payload.provider, payload.host)
    try:
        job_id = await job_manager.submit(
            "stress",
            lambda jid: run_stress_from_request(payload, job=job_manager.jobs[jid]),
            owner=caller,
            request=payload_request(payload, load_settings()),
            progress=initial_stress_progress(payload),
        )
    except TooManyActiveJobsError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    job = job_manager.jobs[job_id]
    return _job_to_response(job)


@router.post("/soak", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_soak(
    request: Request,
    payload: SoakRequest,
    job_manager: JobManager = Depends(get_job_manager),
    caller: str = Depends(require_auth),
) -> JobResponse:
    """Hold concurrency and classify drift. Duration is seconds, max 300."""
    _check_provider_hosts(request, payload.provider, payload.host)
    try:
        job_id = await job_manager.submit(
            "soak",
            lambda jid: run_soak_from_request(payload, job=job_manager.jobs[jid]),
            owner=caller,
            request=payload_request(payload, load_settings()),
            progress=initial_soak_progress(payload),
        )
    except TooManyActiveJobsError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    job = job_manager.jobs[job_id]
    return _job_to_response(job)


@router.get("/jobs", response_model=JobsListResponse)
async def list_jobs(
    job_manager: JobManager = Depends(get_job_manager),
    caller: str = Depends(require_auth),
) -> JobsListResponse:
    """The caller's in-memory API jobs. The result payload is omitted; poll `/jobs/{id}`."""
    return JobsListResponse(
        jobs=[_job_to_summary(job) for job in job_manager.list_jobs() if job.owner == caller]
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
    caller: str = Depends(require_auth),
) -> JobResponse:
    job = job_manager.jobs.get(job_id)
    if job is None or job.owner != caller:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    return _job_to_response(job)


@router.get("/models", response_model=ModelsResponse)
async def get_models(
    request: Request,
    provider: str = "ollama",
    host: str | None = None,
    _: None = Depends(require_auth),
) -> ModelsResponse:
    """List tags on an Ollama or vLLM instance. Does not generate."""
    from atomics.api._discovery import LISTABLE_PROVIDERS

    if provider.lower() in LISTABLE_PROVIDERS:
        _check_provider_hosts(request, provider.lower(), host)
    else:
        _check_payload_hosts(request, ("host", host))
    body = await list_models(provider, host)
    return ModelsResponse(**body)


@router.post("/provider-test", response_model=ProviderTestResponse)
async def provider_test(
    request: Request,
    payload: ProviderTestRequest,
    job_manager: JobManager = Depends(get_job_manager),
    caller: str = Depends(require_auth),
) -> ProviderTestResponse:
    """Health-check a provider and generate a fixed 2+2 probe. Spends tokens."""
    _check_provider_hosts(request, payload.provider, payload.host)
    try:
        job_manager.check_caller_spend(caller)
    except TooManyActiveJobsError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    body = await run_provider_test(payload)
    job_manager.record_spend(caller, float(body.get("cost_usd") or 0.0))
    return ProviderTestResponse(**body)


@router.get("/runs/{run_id}")
async def get_run(
    run_id: str,
    request: Request,
    _: None = Depends(require_auth),
) -> dict[str, object]:
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
    request: Request,
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
    settings = request.app.state.settings
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
    request: Request,
    limit: int = Query(default=10, ge=1, le=100),
    _: None = Depends(require_auth),
) -> ReportResponse:
    settings = request.app.state.settings
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
        request=short_request(job.request),
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
        request=job.request,
        progress=job.progress,
    )
