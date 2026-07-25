from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from atomics.api.dependencies import require_auth, require_worker_auth
from atomics.distributed.coordinator import Coordinator
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
    assignment = coordinator.submit_assignment(
        assignment_id,
        payload.result_json,
        error=payload.error,
    )
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
    if payload.mode != JobMode.SPLIT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only split mode is supported in phase 1",
        )
    if payload.worker_selector:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "worker_selector is not supported in phase 1: split mode assigns "
                "each task to the next available worker. Remove the selector or "
                "wait for fleet mode."
            ),
        )
    if not payload.run_request:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="run_request is required",
        )
    task_specs = _build_task_specs(payload.run_request)
    job = coordinator.create_split_job(payload, task_specs)
    return job


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
