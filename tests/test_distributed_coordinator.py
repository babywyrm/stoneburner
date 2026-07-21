import pytest

from atomics.distributed.coordinator import Coordinator
from atomics.distributed.models import DistributedRunRequest, JobMode, WorkerRegisterRequest
from atomics.storage.schema import init_db


@pytest.fixture
def coordinator(tmp_path):
    db = tmp_path / "test.db"
    conn = init_db(db)
    return Coordinator(conn)


def test_register_worker(coordinator):
    req = WorkerRegisterRequest(labels={"provider": "ollama"})
    w = coordinator.register_worker(req, api_key_hint="1234")
    assert w.labels["provider"] == "ollama"
    assert w.api_key_hint == "1234"
    assert w.status.value == "online"


def test_heartbeat_updates_last_seen(coordinator):
    w = coordinator.register_worker(WorkerRegisterRequest())
    w2 = coordinator.heartbeat(w.worker_id)
    assert w2 is not None
    assert w2.status.value == "online"


def test_create_split_job_creates_assignments(coordinator):
    req = DistributedRunRequest(mode=JobMode.SPLIT, run_request={"iterations": 3})
    job = coordinator.create_split_job(req, [{"i": 1}, {"i": 2}, {"i": 3}])
    assert job.mode == JobMode.SPLIT
    rows = coordinator._conn.execute(
        "SELECT COUNT(*) FROM distributed_assignments WHERE job_id = ?",
        (job.job_id,),
    ).fetchone()
    assert rows[0] == 3


def test_claim_assignment(coordinator):
    job = coordinator.create_split_job(
        DistributedRunRequest(mode=JobMode.SPLIT), [{"i": 1}]
    )
    w = coordinator.register_worker(WorkerRegisterRequest())
    a = coordinator.claim_assignment(w.worker_id)
    assert a is not None
    assert a.worker_id == w.worker_id
    assert a.status.value == "assigned"
    assert a.job_id == job.job_id


def test_submit_assignment_completes_job(coordinator):
    w = coordinator.register_worker(WorkerRegisterRequest())
    job = coordinator.create_split_job(
        DistributedRunRequest(mode=JobMode.SPLIT), [{"i": 1}]
    )
    a = coordinator.claim_assignment(w.worker_id)
    assert a is not None
    coordinator.submit_assignment(a.assignment_id, '{"ok": true}')
    job2 = coordinator.get_job(job.job_id)
    assert job2 is not None
    assert job2.status.value == "completed"
