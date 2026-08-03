from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from atomics.api.dependencies import require_auth, require_worker_auth
from atomics.distributed.coordinator import AssignmentRejectedError, Coordinator
from atomics.distributed.models import (
    DistributedJob,
    DistributedRunRequest,
    JobMode,
    TaskAssignment,
    TaskResultSubmission,
    WorkerRegisterRequest,
    WorkerRegisterResponse,
)
from atomics.models import BurnTier
from atomics.tasks import get_weighted_task

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
    return request.app.state.coordinator


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
) -> dict:
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
    worker_id: str,
    assignment_id: str,
    payload: TaskResultSubmission,
    coordinator: Coordinator = Depends(get_coordinator),
    _: None = Depends(require_worker_auth),
) -> dict:
    try:
        assignment = coordinator.submit_assignment(
            assignment_id,
            payload.result_json,
            worker_id=worker_id,
            error=payload.error,
        )
    except AssignmentRejectedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    if assignment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Assignment not found"
        )
    return {"status": assignment.status.value}


@router.post(
    "/distributed/runs",
    response_model=DistributedJob,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_distributed_run(
    payload: DistributedRunRequest,
    coordinator: Coordinator = Depends(get_coordinator),
    _: None = Depends(require_auth),
) -> DistributedJob:
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
    if not payload.run_request:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="run_request is required",
        )
    if payload.mode is JobMode.FULL:
        task_spec = {"mode": "full", "run_request": payload.run_request, "runtime": "python"}
        if payload.worker_selector:
            try:
                return coordinator.create_full_job_from_selector(
                    payload, payload.worker_selector, task_spec=task_spec
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(exc),
                ) from exc
        return coordinator.create_full_job(payload, [], task_spec=task_spec)
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
        return coordinator.create_fleet_job(payload, task_specs, workers)
    return coordinator.create_split_job(payload, task_specs)


@router.get("/distributed/runs/{job_id}", response_model=DistributedJob)
async def get_job(
    job_id: str,
    coordinator: Coordinator = Depends(get_coordinator),
    _: None = Depends(require_auth),
) -> DistributedJob:
    job = coordinator.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job


@router.get("/distributed/runs")
async def list_jobs(
    limit: int = 20,
    coordinator: Coordinator = Depends(get_coordinator),
    _: None = Depends(require_auth),
) -> dict[str, list[DistributedJob]]:
    """List recent distributed jobs, newest first."""
    jobs = coordinator.list_jobs(limit=max(1, limit))
    return {"jobs": jobs}


@router.get("/workers")
async def list_workers(
    coordinator: Coordinator = Depends(get_coordinator),
    _: None = Depends(require_auth),
) -> dict[str, list[Any]]:
    """List all registered workers."""
    workers = coordinator.list_workers()
    return {"workers": [w.model_dump(mode="json") for w in workers]}
