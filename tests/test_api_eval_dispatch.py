import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from atomics.api._runners import SUPPORTED_EVAL_SUITES
from atomics.api.config import ServerSettings
from atomics.api.server import create_app


@pytest.fixture
def client():
    app = create_app(settings=ServerSettings(no_auth=True))
    with TestClient(app) as tc:
        yield tc


@pytest.mark.asyncio
async def test_post_evals_accuracy(client):
    with patch("atomics.api._runners.run_eval", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = type(
            "S",
            (),
            {
                "overall_accuracy": 0.9,
                "fixtures_run": 3,
                "total_tokens": 100,
                "total_cost_usd": 0.01,
                "fixture_results": [1, 2, 3],
            },
        )()
        resp = client.post(
            "/api/v1/evals", json={"suite": "accuracy", "provider": "ollama"}
        )
    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_post_evals_rag(client):
    with patch("atomics.api._runners.run_rag", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = type(
            "S",
            (),
            {
                "overall_score": 0.8,
                "fixtures_run": 2,
                "total_tokens": 50,
                "total_cost_usd": 0.005,
                "fixture_results": [1, 2],
            },
        )()
        resp = client.post("/api/v1/evals", json={"suite": "rag", "provider": "ollama"})
    assert resp.status_code == 202


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("suite", "runner_attr", "headline_attr"),
    [
        ("refusal", "run_refusal", "calibration_score"),
        ("redblue", "run_redblue", "overall_quality"),
        ("codereview", "run_codereview", "review_score"),
        ("toolcall", "run_toolcall_suite", "dangerous_call_rate"),
    ],
)
async def test_post_evals_security_suite_is_accepted(client, suite, runner_attr, headline_attr):
    summary = type(
        "S",
        (),
        {
            headline_attr: 0.8,
            "fixture_results": [1],
            "fixtures": [1],
            "total_tokens": 10,
            "total_cost_usd": 0.001,
        },
    )()
    with patch(f"atomics.api._runners.{runner_attr}", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = summary
        resp = client.post("/api/v1/evals", json={"suite": suite, "provider": "ollama"})
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        body = None
        for _ in range(80):
            poll = client.get(f"/api/v1/jobs/{job_id}")
            body = poll.json()
            if body["status"] in {"completed", "failed"}:
                break
            time.sleep(0.05)
    assert body is not None
    assert body["status"] == "completed"
    assert body["result"]["suite"] == suite
    assert body["result"]["overall_score"] == 0.8


@pytest.mark.asyncio
async def test_post_evals_unsupported_suite(client):
    resp = client.post(
        "/api/v1/evals", json={"suite": "unknown", "provider": "ollama"}
    )
    assert resp.status_code == 400


def test_supported_eval_suites_cover_the_security_set():
    """An agent using submit_eval should see the same set the CLI security
    family uses. Missing one here is how MCP looked complete while hiding it."""
    assert {
        "adversarial",
        "refusal",
        "redblue",
        "toolcall",
        "codereview",
    }.issubset(SUPPORTED_EVAL_SUITES)
