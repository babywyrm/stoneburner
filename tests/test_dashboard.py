"""Tests for the optional web dashboard."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from atomics.api.config import ServerSettings
from atomics.api.server import create_app


@pytest.fixture
def client_without_dashboard():
    app = create_app(ServerSettings(no_auth=True, with_dashboard=False))
    with TestClient(app) as tc:
        yield tc


@pytest.fixture
def client_with_dashboard(tmp_path):
    app = create_app(
        ServerSettings(no_auth=True, with_dashboard=True, db_path=tmp_path / "dash.db")
    )
    with TestClient(app) as tc:
        yield tc


def test_dashboard_is_unavailable_by_default(client_without_dashboard):
    res = client_without_dashboard.get("/dashboard")
    assert res.status_code == 404


def test_dashboard_returns_html_when_enabled(client_with_dashboard):
    res = client_with_dashboard.get("/dashboard")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "atomics dashboard" in res.text


def test_dashboard_html_is_self_contained(client_with_dashboard):
    res = client_with_dashboard.get("/dashboard")
    # The tag carries a per-response CSP nonce, so match the prefix.
    assert "<script nonce=" in res.text
    assert "Recent runs" in res.text
    assert "Distributed jobs" in res.text
    assert "Workers" in res.text
    assert "Compare by provider" in res.text
    assert "run-detail" in res.text
    assert "/api/v1/runs/" in res.text
    assert "selectRun" in res.text
    assert "Trends" in res.text
    assert "/api/v1/reports/trends" in res.text
    assert "API jobs" in res.text
    assert "selectJob" in res.text
    assert "loadJobDetail" in res.text
    script = res.text.split("<script", 1)[1]
    assert "data.result" not in script


def test_dashboard_run_detail_omits_result_json(tmp_path):
    from atomics.storage.records import EvaluationResultRecord
    from atomics.storage.repository import MetricsRepository

    db = tmp_path / "dash-detail.db"
    repo = MetricsRepository(db)
    repo.create_run("dash-run", tier="refusal", provider="ollama", model="qwen")
    repo.save_evaluation_result(
        EvaluationResultRecord(
            run_id="dash-run",
            suite="refusal",
            fixture_id="rf-01",
            status="complete",
            generation_status="completed",
            judge_status="scored",
            latency_ms=12.0,
            input_tokens=3,
            output_tokens=2,
            total_tokens=5,
            score=0.75,
            result_json={"prompt": "do not leak"},
            provider="ollama",
            model="qwen",
        )
    )
    repo.close()

    app = create_app(ServerSettings(no_auth=True, with_dashboard=True, db_path=db))
    with TestClient(app) as tc:
        page = tc.get("/dashboard")
        assert "selectRun" in page.text
        detail = tc.get("/api/v1/runs/dash-run")
    assert detail.status_code == 200
    assert detail.json()["fixtures"][0]["id"] == "rf-01"
    assert "do not leak" not in detail.text


def test_dashboard_lists_no_data_when_empty(client_with_dashboard):
    res = client_with_dashboard.get("/api/v1/distributed/runs")
    assert res.status_code == 200
    assert res.json() == {"jobs": []}


def test_dashboard_trends_use_the_server_database(tmp_path):
    from atomics.storage.records import EvaluationResultRecord
    from atomics.storage.repository import MetricsRepository

    db = tmp_path / "dash-trends.db"
    repo = MetricsRepository(db)
    repo.create_run("dash-trend", tier="refusal", provider="ollama", model="qwen")
    repo.save_evaluation_result(
        EvaluationResultRecord(
            run_id="dash-trend",
            suite="refusal",
            fixture_id="rf-01",
            status="complete",
            generation_status="completed",
            judge_status="scored",
            latency_ms=5.0,
            input_tokens=2,
            output_tokens=3,
            total_tokens=5,
            result_json={"prompt": "dash-trend-secret"},
            provider="ollama",
            model="qwen",
        )
    )
    repo.close()

    app = create_app(ServerSettings(no_auth=True, with_dashboard=True, db_path=db))
    with TestClient(app) as tc:
        resp = tc.get("/api/v1/reports/trends?hours=24")
    assert resp.status_code == 200
    assert resp.json()["rows"][0]["total_tokens"] == 5
    assert "dash-trend-secret" not in resp.text


@pytest.mark.parametrize(
    "path",
    ["/api/v1/distributed/runs", "/api/v1/workers", "/api/v1/jobs", "/api/v1/reports/trends"],
)
def test_dashboard_data_endpoints_require_auth_when_not_no_auth(path):
    app = create_app(ServerSettings(no_auth=False, api_keys={"secret"}, with_dashboard=True))
    with TestClient(app) as tc:
        res = tc.get(path)
        assert res.status_code == 401

        res = tc.get(path, headers={"X-API-Key": "secret"})
        assert res.status_code == 200
