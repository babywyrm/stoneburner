from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from atomics.api.config import ServerSettings
from atomics.api.server import create_app


@pytest.fixture
def client():
    app = create_app(settings=ServerSettings(no_auth=True))
    with TestClient(app) as tc:
        yield tc


@pytest.mark.asyncio
async def test_post_runs_creates_job(client):
    with patch(
        "atomics.api.routes.run_benchmark_from_request", new_callable=AsyncMock
    ) as mock_run:
        mock_run.return_value = {"run_id": "abc123", "tasks": 3, "success": 3}
        resp = client.post("/api/v1/runs", json={"provider": "ollama"})
        assert resp.status_code == 202
        body = resp.json()
        assert body["kind"] == "run"
        assert body["status"] in ("pending", "running", "completed")
        job_id = body["job_id"]

        # Keep the mock patched while the background job finishes.
        for _ in range(50):
            resp = client.get(f"/api/v1/jobs/{job_id}")
            if resp.json()["status"] == "completed":
                break
        assert resp.json()["status"] == "completed"
        assert resp.json()["result"]["run_id"] == "abc123"


@pytest.mark.asyncio
async def test_post_evals_creates_job(client):
    with patch(
        "atomics.api.routes.run_eval_suite", new_callable=AsyncMock
    ) as mock_run:
        mock_run.return_value = {
            "suite": "accuracy",
            "overall_accuracy": 0.85,
            "fixtures_run": 5,
        }
        resp = client.post(
            "/api/v1/evals", json={"suite": "accuracy", "provider": "ollama"}
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["kind"] == "eval"
    assert body["status"] in ("pending", "running", "completed")

@pytest.mark.asyncio
async def test_poll_slow_job_returns_running(client):
    import asyncio
    import threading

    started = threading.Event()
    release = threading.Event()

    async def slow_run(_payload):
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.01)
        return {"run_id": "slow-job", "tasks": 1, "success": 1}

    with patch(
        "atomics.api.routes.run_benchmark_from_request",
        new=AsyncMock(side_effect=slow_run),
    ):
        resp = client.post("/api/v1/runs", json={"provider": "ollama"})
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        assert started.wait(timeout=2.0)

        poll = client.get(f"/api/v1/jobs/{job_id}")
        assert poll.status_code == 200
        assert poll.json()["status"] == "running"

        release.set()

        for _ in range(50):
            poll = client.get(f"/api/v1/jobs/{job_id}")
            if poll.json()["status"] == "completed":
                break
        assert poll.json()["status"] == "completed"
        assert poll.json()["result"]["run_id"] == "slow-job"

