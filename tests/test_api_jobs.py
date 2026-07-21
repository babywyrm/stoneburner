import asyncio

import pytest

from atomics.api.jobs import JobManager, JobStatus


@pytest.mark.asyncio
async def test_job_manager_lifecycle():
    manager = JobManager()

    async def work(job_id):
        await asyncio.sleep(0.01)
        return {"ok": True}

    job_id = await manager.submit("run", work)
    assert job_id in manager.jobs
    assert manager.jobs[job_id].status == JobStatus.PENDING

    await manager.wait_for(job_id, timeout=1.0)
    assert manager.jobs[job_id].status == JobStatus.COMPLETED
    assert manager.jobs[job_id].result == {"ok": True}


@pytest.mark.asyncio
async def test_job_manager_failure():
    manager = JobManager()

    async def work(job_id):
        raise ValueError("boom")

    job_id = await manager.submit("run", work)
    await manager.wait_for(job_id, timeout=1.0)
    assert manager.jobs[job_id].status == JobStatus.FAILED
    assert manager.jobs[job_id].error is not None
    assert manager.jobs[job_id].error["message"] == "boom"


@pytest.mark.asyncio
async def test_wait_for_timeout_zero_does_not_cancel():
    manager = JobManager()
    started = asyncio.Event()

    async def work(job_id):
        started.set()
        await asyncio.sleep(0.2)
        return {"ok": True}

    job_id = await manager.submit("run", work)
    await started.wait()

    job = await manager.wait_for(job_id, timeout=0.0)
    assert job.status == JobStatus.RUNNING
    assert job._task is not None
    assert not job._task.done()
    assert not job._task.cancelled()

    await manager.wait_for(job_id, timeout=1.0)
    assert manager.jobs[job_id].status == JobStatus.COMPLETED
    assert manager.jobs[job_id].result == {"ok": True}


@pytest.mark.asyncio
async def test_wait_for_positive_timeout_does_not_cancel():
    manager = JobManager()
    started = asyncio.Event()

    async def work(job_id):
        started.set()
        await asyncio.sleep(0.2)
        return {"ok": True}

    job_id = await manager.submit("run", work)
    await started.wait()

    job = await manager.wait_for(job_id, timeout=0.01)
    assert job.status == JobStatus.RUNNING
    assert job._task is not None
    assert not job._task.cancelled()

    await manager.wait_for(job_id, timeout=1.0)
    assert manager.jobs[job_id].status == JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_wait_for_missing_job_raises_key_error():
    manager = JobManager()
    with pytest.raises(KeyError):
        await manager.wait_for("missing-id")


@pytest.mark.asyncio
async def test_wait_for_completed_job_returns_immediately():
    manager = JobManager()

    async def work(job_id):
        return {"done": True}

    job_id = await manager.submit("run", work)
    await manager.wait_for(job_id, timeout=1.0)
    job = await manager.wait_for(job_id)
    assert job.status == JobStatus.COMPLETED
    assert job.result == {"done": True}


@pytest.mark.asyncio
async def test_wait_for_no_timeout_awaits_task():
    manager = JobManager()

    async def work(job_id):
        await asyncio.sleep(0.05)
        return {"ok": True}

    job_id = await manager.submit("run", work)
    job = await manager.wait_for(job_id)  # timeout=None
    assert job.status == JobStatus.COMPLETED
    assert job.result == {"ok": True}

