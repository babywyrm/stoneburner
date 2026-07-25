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
async def test_end_to_end_fleet_run(client):
    """Two hosts, one identical task set each, one comparable rollup.

    Drives the real HTTP routes rather than the coordinator directly, so the whole
    path is covered: label-matched broadcast, per-host pinning enforced at claim
    time, and the per-worker summary a completed fleet run must produce.
    """
    fast = client.post(
        "/api/v1/workers/register", json={"labels": {"gpu": "4090", "site": "lab"}}
    ).json()["worker_id"]
    slow = client.post(
        "/api/v1/workers/register", json={"labels": {"gpu": "3060", "site": "lab"}}
    ).json()["worker_id"]
    # Present but unselected, to prove the selector actually narrows the fleet.
    other_site = client.post(
        "/api/v1/workers/register", json={"labels": {"gpu": "4090", "site": "desk"}}
    ).json()["worker_id"]

    run_resp = client.post(
        "/api/v1/distributed/runs",
        json={
            "mode": "fleet",
            "run_request": {"iterations": 2, "tier": "ez"},
            "worker_selector": {"site": "lab"},
        },
    )
    assert run_resp.status_code == 202
    job_id = run_resp.json()["job_id"]

    assert client.get(f"/api/v1/workers/{other_site}/jobs/next").json() is None

    prompts_seen: dict[str, list[str]] = {fast: [], slow: []}
    for worker_id in (fast, slow):
        for _ in range(2):
            poll = client.get(f"/api/v1/workers/{worker_id}/jobs/next")
            assert poll.status_code == 200
            body = poll.json()
            assert body is not None, f"{worker_id} ran out of work early"
            assignment = TaskAssignment(**body)
            assert assignment.target_worker_id == worker_id
            prompts_seen[worker_id].append(assignment.task_spec["prompt"])

            with patch(
                "atomics.distributed.worker_runner.make_provider",
                return_value=FakeProvider(),
            ):
                result = await execute_assignment(
                    assignment, provider_name="fake", model="fake-model"
                )
            submit = client.post(
                f"/api/v1/workers/{worker_id}/jobs/{assignment.assignment_id}/result",
                json={"status": "completed", "result_json": json.dumps(result)},
            )
            assert submit.status_code == 200
        # Its slice is done, and it must not start eating the other host's work.
        assert client.get(f"/api/v1/workers/{worker_id}/jobs/next").json() is None

    assert sorted(prompts_seen[fast]) == sorted(prompts_seen[slow])

    finished = client.get(f"/api/v1/distributed/runs/{job_id}").json()
    assert finished["status"] == "completed"
    summary = json.loads(finished["summary_json"])
    by_id = {w["worker_id"]: w for w in summary["workers"]}
    assert set(by_id) == {fast, slow}
    assert by_id[fast]["completed"] == 2
    assert by_id[slow]["completed"] == 2
    assert by_id[fast]["labels"]["gpu"] == "4090"
    assert summary["completed"] == 4


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
        "atomics.distributed.worker_runner.make_provider",
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
