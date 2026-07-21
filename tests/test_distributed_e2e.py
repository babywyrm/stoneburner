"""End-to-end local distributed run: register → claim → execute → submit → complete."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from atomics.api.server import create_app
from atomics.distributed.models import TaskAssignment
from atomics.distributed.worker_runner import execute_assignment
from atomics.providers.base import BaseProvider, ProviderResponse


@pytest.fixture
def client(tmp_path):
    app = create_app(no_auth=True, db_path=tmp_path / "distributed_e2e.db")
    with TestClient(app) as tc:
        yield tc


class FakeProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "fake"

    @property
    def default_model(self) -> str | None:
        return "fake-model"

    async def generate(self, prompt, **kwargs):
        return ProviderResponse(
            text="fake result",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            model="fake-model",
            estimated_cost_usd=0.0,
            latency_ms=100.0,
        )

    async def health_check(self):
        return True


@pytest.mark.asyncio
async def test_end_to_end_split_run(client):
    # 1. Register a worker
    reg = client.post("/api/v1/workers/register", json={"labels": {"provider": "fake"}})
    assert reg.status_code == 200
    worker_id = reg.json()["worker_id"]

    # 2. Submit a split run with 1 iteration
    run_resp = client.post(
        "/api/v1/distributed/runs",
        json={"mode": "split", "run_request": {"iterations": 1, "tier": "ez"}},
    )
    assert run_resp.status_code == 202
    job_id = run_resp.json()["job_id"]

    # 3. Worker polls for an assignment
    poll = client.get(f"/api/v1/workers/{worker_id}/jobs/next")
    assert poll.status_code == 200
    poll_body = poll.json()
    assert poll_body is not None
    assignment = TaskAssignment(**poll_body)
    assert assignment.job_id == job_id
    assert assignment.task_spec.get("task_name")
    assert assignment.task_spec.get("prompt")

    # 4. Execute the assignment with a mocked provider (no network)
    with patch(
        "atomics.distributed.worker_runner._make_provider",
        return_value=FakeProvider(),
    ):
        result = await execute_assignment(assignment, provider_name="fake", model="fake-model")

    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert result["response"] == "fake result"
    assert result["run_id"] == job_id

    # 5. Submit the result
    submit = client.post(
        f"/api/v1/workers/{worker_id}/jobs/{assignment.assignment_id}/result",
        json={"status": "completed", "result_json": json.dumps(result)},
    )
    assert submit.status_code == 200
    assert submit.json()["status"] == "completed"

    # 6. Verify job completed
    status_resp = client.get(f"/api/v1/distributed/runs/{job_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "completed"
