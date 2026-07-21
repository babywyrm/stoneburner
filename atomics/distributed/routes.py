from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from atomics.api.auth import AuthBackend
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


def _build_task_specs(run_request: dict) -> list[dict]:
    """Build one catalog-backed task spec per iteration for split runs."""
    tier = _parse_tier(run_request.get("tier", "baseline"))
    iterations = int(run_request.get("iterations", 1))
    specs: list[dict] = []
    for _ in range(max(iterations, 0)):
        task, prompt = get_weighted_task(tier)
        specs.append(
            {
                "task_name": task.name,
                "prompt": prompt,
                "category": task.category.value,
                "complexity": task.complexity.value,
                "max_output_tokens": task.max_output_tokens,
            }
        )
    return specs


def get_coordinator(request: Request) -> Coordinator:
    return request.app.state.coordinator


def get_worker_auth(request: Request) -> AuthBackend:
    return request.app.state.worker_auth


async def require_worker_auth(
    request: Request, auth: AuthBackend = Depends(get_worker_auth)
) -> None:
    if not await auth.authenticate(request):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid worker API key"
        )


@router.post("/workers/register", response_model=WorkerRegisterResponse)
async def register_worker(
    payload: WorkerRegisterRequest,
    coordinator: Coordinator = Depends(get_coordinator),
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
) -> DistributedJob:
    if payload.mode != JobMode.SPLIT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only split mode is supported in phase 1",
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
) -> DistributedJob:
    job = coordinator.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job
