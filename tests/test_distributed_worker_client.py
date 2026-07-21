"""Tests for distributed worker client polling loop."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atomics.distributed.worker_client import WorkerClient


def _mock_response(payload, *, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture
def mock_httpx_client():
    client = AsyncMock()
    with patch(
        "atomics.distributed.worker_client.httpx.AsyncClient",
        return_value=client,
    ):
        yield client


@pytest.mark.asyncio
async def test_register_stores_worker_id(mock_httpx_client):
    mock_httpx_client.post.return_value = _mock_response({"worker_id": "w-123"})

    worker = WorkerClient(
        coordinator_url="http://coordinator:8000",
        api_key="secret",
        labels={"provider": "ollama"},
        endpoint="http://worker:9000",
    )
    await worker.register()

    assert worker._worker_id == "w-123"
    mock_httpx_client.post.assert_awaited_once()
    call = mock_httpx_client.post.await_args
    assert call.args[0] == "http://coordinator:8000/api/v1/workers/register"
    assert call.kwargs["json"]["labels"] == {"provider": "ollama"}
    assert call.kwargs["json"]["endpoint"] == "http://worker:9000"
    await worker.close()


@pytest.mark.asyncio
async def test_heartbeat_posts_to_worker_url(mock_httpx_client):
    mock_httpx_client.post.return_value = _mock_response({"status": "ok"})

    worker = WorkerClient(coordinator_url="http://coordinator:8000/", api_key="secret")
    worker._worker_id = "w-abc"
    await worker.heartbeat()

    mock_httpx_client.post.assert_awaited_once_with(
        "http://coordinator:8000/api/v1/workers/w-abc/heartbeat"
    )
    await worker.close()


@pytest.mark.asyncio
async def test_poll_and_execute_posts_completed_result(mock_httpx_client):
    assignment = {
        "assignment_id": "a-1",
        "job_id": "j-1",
        "task_spec": {"task_name": "quick_question", "prompt": "hi"},
    }
    mock_httpx_client.get.return_value = _mock_response(assignment)
    mock_httpx_client.post.return_value = _mock_response({"status": "completed"})

    executor = AsyncMock(return_value={"ok": True, "score": 1})
    worker = WorkerClient(
        coordinator_url="http://coordinator:8000",
        api_key="secret",
        executor=executor,
    )
    worker._worker_id = "w-1"

    worked = await worker.poll_and_execute()

    assert worked is True
    executor.assert_awaited_once()
    assert executor.await_args.args[0].assignment_id == "a-1"

    mock_httpx_client.get.assert_awaited_once_with(
        "http://coordinator:8000/api/v1/workers/w-1/jobs/next"
    )
    mock_httpx_client.post.assert_awaited_once()
    post = mock_httpx_client.post.await_args
    assert (
        post.args[0]
        == "http://coordinator:8000/api/v1/workers/w-1/jobs/a-1/result"
    )
    assert post.kwargs["json"]["status"] == "completed"
    assert json.loads(post.kwargs["json"]["result_json"]) == {"ok": True, "score": 1}
    await worker.close()


@pytest.mark.asyncio
async def test_poll_and_execute_returns_false_when_empty(mock_httpx_client):
    mock_httpx_client.get.return_value = _mock_response(None)

    worker = WorkerClient(coordinator_url="http://coordinator:8000", api_key="secret")
    worker._worker_id = "w-1"

    worked = await worker.poll_and_execute()

    assert worked is False
    mock_httpx_client.post.assert_not_awaited()
    await worker.close()


@pytest.mark.asyncio
async def test_poll_and_execute_reports_failure(mock_httpx_client):
    assignment = {
        "assignment_id": "a-2",
        "job_id": "j-2",
        "task_spec": {"task_name": "quick_question", "prompt": "boom"},
    }
    mock_httpx_client.get.return_value = _mock_response(assignment)
    mock_httpx_client.post.return_value = _mock_response({"status": "failed"})

    executor = AsyncMock(side_effect=RuntimeError("boom"))
    worker = WorkerClient(
        coordinator_url="http://coordinator:8000",
        api_key="secret",
        executor=executor,
    )
    worker._worker_id = "w-1"

    worked = await worker.poll_and_execute()

    assert worked is True
    post = mock_httpx_client.post.await_args
    assert post.kwargs["json"]["status"] == "failed"
    assert "boom" in post.kwargs["json"]["error"]
    await worker.close()
