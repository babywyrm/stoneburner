"""Tests for Coordinator listing helpers used by the dashboard."""

from __future__ import annotations

from atomics.distributed.coordinator import Coordinator
from atomics.distributed.models import (
    DistributedRunRequest,
    JobMode,
    WorkerRegisterRequest,
)
from atomics.storage.schema import init_db


def test_list_workers_returns_newest_first(tmp_path):
    conn = init_db(tmp_path / "test.db")
    coordinator = Coordinator(conn)
    w1 = coordinator.register_worker(WorkerRegisterRequest())
    w2 = coordinator.register_worker(WorkerRegisterRequest())
    workers = coordinator.list_workers()
    assert [w.worker_id for w in workers] == [w2.worker_id, w1.worker_id]


def test_list_jobs_returns_newest_first(tmp_path):
    conn = init_db(tmp_path / "test.db")
    coordinator = Coordinator(conn)
    coordinator.create_split_job(DistributedRunRequest(mode=JobMode.SPLIT), [{"task_name": "a"}])
    coordinator.create_split_job(DistributedRunRequest(mode=JobMode.SPLIT), [{"task_name": "b"}])
    jobs = coordinator.list_jobs(limit=5)
    assert len(jobs) == 2
    assert jobs[0].mode == jobs[1].mode == JobMode.SPLIT
