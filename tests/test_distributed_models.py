from atomics.distributed.models import (
    AssignmentStatus,
    DistributedJob,
    JobMode,
    JobStatus,
    TaskAssignment,
    Worker,
    WorkerStatus,
)


def test_worker_defaults():
    w = Worker(worker_id="w-1", labels={"provider": "ollama"})
    assert w.status == WorkerStatus.ONLINE
    assert w.worker_id == "w-1"
    assert w.labels["provider"] == "ollama"


def test_distributed_job_defaults():
    j = DistributedJob(job_id="j-1", mode=JobMode.SPLIT, request_json="{}")
    assert j.status == JobStatus.PENDING


def test_task_assignment_defaults():
    a = TaskAssignment(assignment_id="a-1", job_id="j-1", task_spec={"x": 1})
    assert a.status == AssignmentStatus.PENDING
