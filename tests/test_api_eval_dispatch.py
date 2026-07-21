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
async def test_post_evals_unsupported_suite(client):
    resp = client.post(
        "/api/v1/evals", json={"suite": "unknown", "provider": "ollama"}
    )
    assert resp.status_code == 400
