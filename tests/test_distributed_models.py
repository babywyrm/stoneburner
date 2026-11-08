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
    assert w.registered_at.tzinfo is not None


def test_distributed_job_defaults():
    j = DistributedJob(job_id="j-1", mode=JobMode.SPLIT, request_json="{}")
    assert j.status == JobStatus.PENDING


def test_task_assignment_defaults():
    a = TaskAssignment(assignment_id="a-1", job_id="j-1", task_spec={"x": 1})
    assert a.status == AssignmentStatus.PENDING


def test_distributed_run_request_caps_iterations_and_timeout():
    import pytest
    from pydantic import ValidationError

    from atomics.distributed.models import DistributedRunRequest

    with pytest.raises(ValidationError):
        DistributedRunRequest(timeout_seconds=0)
    with pytest.raises(ValidationError):
        DistributedRunRequest(max_retries=20)
    with pytest.raises(ValidationError, match="iterations"):
        DistributedRunRequest(run_request={"iterations": 5000})
    with pytest.raises(ValidationError, match="iterations"):
        DistributedRunRequest(run_request={"iterations": 0})
    with pytest.raises(ValidationError, match="interval"):
        DistributedRunRequest(run_request={"interval": 10000})
    ok = DistributedRunRequest(run_request={"iterations": 10, "interval": 5})
    assert ok.run_request == {"iterations": 10, "interval": 5}
