from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from atomics.api.config import DisallowedHostError
from atomics.api.dependencies import require_auth, require_worker_auth
from atomics.api.jobs import CallerSpendQuotaExceededError, JobManager
from atomics.distributed.coordinator import AssignmentRejectedError, Coordinator
from atomics.distributed.models import (
    AssignmentStatus,
    DistributedJob,
    DistributedRunRequest,
    JobMode,
    TaskAssignment,
    TaskResultSubmission,
    WorkerRegisterRequest,
    WorkerRegisterResponse,
)
from atomics.distributed.rollup import assignment_cost_usd
from atomics.models import BurnTier
from atomics.tasks import get_weighted_task
from atomics.validation import validate_endpoint_url

router = APIRouter(prefix="/api/v1")


def _parse_tier(value: object) -> BurnTier:
    if isinstance(value, BurnTier):
        return value
    try:
        return BurnTier(str(value))
    except ValueError:
        return BurnTier.BASELINE


def _pinned_execution(run_request: dict[str, Any]) -> dict[str, str]:
    """Return the provider/model the submitter pinned for this run, if any.

    Workers default to their own locally configured provider. When the run
    request names one, it travels with every task spec so the work executes
    where the submitter asked rather than wherever it happens to land.
    """
    pinned: dict[str, str] = {}
    for key in ("provider", "model"):
        value = run_request.get(key)
        if isinstance(value, str) and value:
            pinned[key] = value
    return pinned


def _build_task_specs(run_request: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one catalog-backed task spec per iteration for split runs."""
    tier = _parse_tier(run_request.get("tier", "baseline"))
    iterations = int(run_request.get("iterations", 1))
    pinned = _pinned_execution(run_request)
    runtime = str(run_request.get("runtime", "python"))
    specs: list[dict[str, Any]] = []
    for _ in range(max(iterations, 0)):
        task, prompt = get_weighted_task(tier)
        specs.append(
            {
                "task_name": task.name,
                "prompt": prompt,
                "category": task.category.value,
                "complexity": task.complexity.value,
                "max_output_tokens": task.max_output_tokens,
                "runtime": runtime,
                **pinned,
            }
        )
    return specs


def get_coordinator(request: Request) -> Coordinator:
    coordinator = request.app.state.coordinator
    if not isinstance(coordinator, Coordinator):
        raise RuntimeError("Distributed coordinator is not initialized")
    return coordinator


@router.post("/workers/register", response_model=WorkerRegisterResponse)
async def register_worker(
    payload: WorkerRegisterRequest,
    coordinator: Coordinator = Depends(get_coordinator),
    _: None = Depends(require_worker_auth),
) -> WorkerRegisterResponse:
    worker = coordinator.register_worker(payload)
    return WorkerRegisterResponse(worker_id=worker.worker_id)


@router.post("/workers/{worker_id}/heartbeat")
async def heartbeat(
    worker_id: str,
    coordinator: Coordinator = Depends(get_coordinator),
    _: None = Depends(require_worker_auth),
) -> dict[str, str]:
    worker = coordinator.heartbeat(worker_id)
    if worker is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Worker not found")
    return {"status": "ok"}


@router.get("/workers/{worker_id}/jobs/next", response_model=TaskAssignment | None)
async def next_assignment(
    worker_id: str,
    coordinator: Coordinator = Depends(get_coordinator),
    _: None = Depends(require_worker_auth),
) -> TaskAssignment | None:
    coordinator.heartbeat(worker_id)
    return coordinator.claim_assignment(worker_id)


@router.post("/workers/{worker_id}/jobs/{assignment_id}/result")
async def submit_result(
    request: Request,
    worker_id: str,
    assignment_id: str,
    payload: TaskResultSubmission,
    coordinator: Coordinator = Depends(get_coordinator),
    _: None = Depends(require_worker_auth),
) -> dict[str, str]:
    try:
        assignment = coordinator.submit_assignment(
            assignment_id,
            payload.result_json,
            worker_id=worker_id,
            error=payload.error,
            failed=payload.status == AssignmentStatus.FAILED,
        )
    except AssignmentRejectedError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if assignment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assignment not found")
    cost = assignment_cost_usd(payload.result_json)
    if cost > 0.0:
        owner = coordinator.job_owner(assignment.job_id)
        job_manager = getattr(request.app.state, "job_manager", None)
        if owner is not None and isinstance(job_manager, JobManager):
            job_manager.record_spend(owner, cost)
    return {"status": assignment.status.value}


def _check_distributed_hosts(request: Request, payload: DistributedRunRequest) -> None:
    settings = request.app.state.settings
    sections: tuple[tuple[str, dict[str, Any] | None, tuple[str, ...]], ...] = (
        ("run_request", payload.run_request, ("host", "ollama_host", "vllm_host", "gateway_url")),
        ("eval_request", payload.eval_request, ("host", "judge_host")),
    )
    for prefix, mapping, keys in sections:
        if not mapping:
            continue
        for key in keys:
            value = mapping.get(key)
            if value is None:
                continue
            label = f"{prefix}.{key}"
            if not isinstance(value, str):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"{label} must be a URL string",
                )
            try:
                validate_endpoint_url(value, label=label)
                settings.check_outbound_host(value, label=label)
            except DisallowedHostError as exc:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
                ) from exc
        if prefix == "eval_request" and mapping.get("profile") is not None:
            from atomics.load.profiles import load_confined_profile

            raw_prof = mapping.get("profile")
            if not isinstance(raw_prof, str):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="eval_request.profile must be a string",
                )
            try:
                prof = load_confined_profile(raw_prof)
                prof_url = prof.ollama_host if prof.type == "ollama" else prof.http_url
                settings.check_outbound_host(prof_url, label="eval_request.profile")
            except DisallowedHostError as exc:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
            except (ValueError, OSError) as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
                ) from exc
        if settings.allowed_hosts is not None:
            provider = mapping.get("provider")
            if isinstance(provider, str) and provider:
                from atomics.config import load_settings
                from atomics.providers.factory import ProviderConfigError, resolve_provider_endpoint

                explicit = next(
                    (
                        mapping[k]
                        for k in keys
                        if k != "judge_host" and isinstance(mapping.get(k), str) and mapping[k]
                    ),
                    None,
                )
                try:
                    eff = resolve_provider_endpoint(
                        provider,
                        explicit,
                        load_settings(),
                        host_label=f"{prefix}.host",
                    )
                    settings.check_outbound_host(eff, label=f"{prefix}.host")
                except DisallowedHostError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)
                    ) from exc
                except (ProviderConfigError, ValueError) as exc:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
                    ) from exc


def _validate_distributed_eval_request(
    request: Request, eval_request: dict[str, Any]
) -> dict[str, Any]:
    """Validate `eval_request` upfront against `EvalRequest` and allowed hosts."""
    from pydantic import ValidationError

    from atomics.api._runners import validate_eval_suite
    from atomics.api.models import EvalRequest
    from atomics.config import load_settings
    from atomics.providers.factory import ProviderConfigError, resolve_provider_endpoint

    candidate = dict(eval_request)
    if not candidate.get("provider"):
        candidate["provider"] = "ollama"
    try:
        parsed = EvalRequest(**candidate)
        suite = validate_eval_suite(parsed.suite)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid eval_request: {exc.errors()[0].get('msg', str(exc))}",
        ) from exc
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

    settings = request.app.state.settings
    if settings.allowed_hosts is not None and not eval_request.get("judge_host"):
        fallback_judge_host = parsed.host if parsed.provider == "ollama" and parsed.host else None
        try:
            judge_eff = resolve_provider_endpoint(
                "ollama",
                fallback_judge_host,
                load_settings(),
                host_label="eval_request.judge_host",
            )
            settings.check_outbound_host(judge_eff, label="eval_request.judge_host")
        except DisallowedHostError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        except (ProviderConfigError, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return dict(eval_request)


@router.post(
    "/distributed/runs",
    response_model=DistributedJob,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_distributed_run(
    request: Request,
    payload: DistributedRunRequest,
    coordinator: Coordinator = Depends(get_coordinator),
    caller: str = Depends(require_auth),
) -> DistributedJob:
    _check_distributed_hosts(request, payload)
    if payload.mode not in (JobMode.SPLIT, JobMode.FLEET, JobMode.FULL):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported mode {payload.mode.value!r}. Use 'split' to divide "
                "work across workers, 'fleet' to run it on every matching worker, "
                "or 'full' to delegate an entire run to one worker."
            ),
        )
    if payload.mode is JobMode.SPLIT and payload.worker_selector:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "worker_selector is not honored in split mode: each task goes to "
                "the next available worker. Drop the selector, or use fleet/full mode "
                "to target workers by label."
            ),
        )
    if payload.run_request and payload.eval_request:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pass either run_request or eval_request, not both",
        )
    if not payload.run_request and not payload.eval_request:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="run_request or eval_request is required",
        )
    job_manager = getattr(request.app.state, "job_manager", None)
    settings = request.app.state.settings
    in_mem_active = job_manager.active_count if isinstance(job_manager, JobManager) else 0
    total_active = in_mem_active + coordinator.active_job_count()
    if total_active >= settings.max_active_jobs:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"{total_active} jobs already running (limit {settings.max_active_jobs})",
        )
    in_mem_caller = (
        job_manager.active_count_for(caller) if isinstance(job_manager, JobManager) else 0
    )
    caller_active = in_mem_caller + coordinator.active_job_count_for(caller)
    if caller_active >= settings.max_active_jobs_per_caller:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"caller already has {caller_active} jobs running "
                f"(per-caller limit {settings.max_active_jobs_per_caller})"
            ),
        )
    if isinstance(job_manager, JobManager):
        try:
            job_manager.check_caller_spend(caller)
        except CallerSpendQuotaExceededError as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)
            ) from exc

    if payload.eval_request is not None:
        if payload.mode is JobMode.SPLIT:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="eval_request requires 'full' or 'fleet' mode",
            )
        validated_eval = _validate_distributed_eval_request(request, payload.eval_request)
        eval_spec: dict[str, Any] = {
            "mode": "full",
            "eval_request": validated_eval,
            "runtime": "python",
        }
        if payload.mode is JobMode.FULL:
            if payload.worker_selector:
                try:
                    return coordinator.create_full_job_from_selector(
                        payload, payload.worker_selector, task_spec=eval_spec, owner=caller
                    )
                except ValueError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=str(exc),
                    ) from exc
            return coordinator.create_full_job(payload, [], task_spec=eval_spec, owner=caller)
        workers = coordinator.matching_workers(payload.worker_selector)
        if not workers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "No online workers match the selector "
                    f"{payload.worker_selector or {}}. Register a worker with "
                    "those labels, or omit the selector to broadcast to every "
                    "online worker."
                ),
            )
        return coordinator.create_fleet_job(payload, [eval_spec], workers, owner=caller)

    assert payload.run_request is not None
    if payload.mode is JobMode.FULL:
        task_spec = {"mode": "full", "run_request": payload.run_request, "runtime": "python"}
        if payload.worker_selector:
            try:
                return coordinator.create_full_job_from_selector(
                    payload, payload.worker_selector, task_spec=task_spec, owner=caller
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(exc),
                ) from exc
        return coordinator.create_full_job(payload, [], task_spec=task_spec, owner=caller)
    # Built once and shared across workers in fleet mode: each spec is sampled
    # randomly, so per-worker generation would give each host different prompts.
    task_specs = _build_task_specs(payload.run_request)
    if payload.mode is JobMode.FLEET:
        workers = coordinator.matching_workers(payload.worker_selector)
        if not workers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "No online workers match the selector "
                    f"{payload.worker_selector or {}}. Register a worker with "
                    "those labels, or omit the selector to broadcast to every "
                    "online worker."
                ),
            )
        return coordinator.create_fleet_job(payload, task_specs, workers, owner=caller)
    return coordinator.create_split_job(payload, task_specs, owner=caller)


@router.get("/distributed/runs/{job_id}", response_model=DistributedJob)
async def get_job(
    job_id: str,
    coordinator: Coordinator = Depends(get_coordinator),
    caller: str = Depends(require_auth),
) -> DistributedJob:
    job = coordinator.get_job(job_id, owner=caller)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job


@router.get("/distributed/runs")
async def list_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    coordinator: Coordinator = Depends(get_coordinator),
    caller: str = Depends(require_auth),
) -> dict[str, list[DistributedJob]]:
    """List the caller's recent distributed jobs, newest first."""
    jobs = coordinator.list_jobs(limit=limit, owner=caller)
    return {"jobs": jobs}


@router.get("/workers")
async def list_workers(
    coordinator: Coordinator = Depends(get_coordinator),
    _: None = Depends(require_auth),
) -> dict[str, list[Any]]:
    """List all registered workers."""
    workers = coordinator.list_workers()
    return {"workers": [w.model_dump(mode="json") for w in workers]}
