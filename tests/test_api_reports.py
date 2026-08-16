from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from atomics.api.config import ServerSettings
from atomics.api.server import create_app


@pytest.fixture
def client():
    app = create_app(settings=ServerSettings(no_auth=True))
    with TestClient(app) as tc:
        yield tc


def test_compare_empty(client):
    with patch("atomics.api.routes.MetricsRepository") as mock_repo:
        mock_repo.return_value.compare_providers.return_value = []
        mock_repo.return_value.close = lambda: None
        resp = client.get("/api/v1/compare?by=provider")
    assert resp.status_code == 200
    assert resp.json() == {"by": "provider", "rows": []}


def test_reports_recent_runs(client):
    with patch("atomics.api.routes.MetricsRepository") as mock_repo:
        mock_repo.return_value.get_recent_runs.return_value = []
        mock_repo.return_value.close = lambda: None
        resp = client.get("/api/v1/reports/recent-runs?limit=5")
    assert resp.status_code == 200
    assert resp.json() == {"runs": []}


def test_compare_invalid_by(client):
    resp = client.get("/api/v1/compare?by=nope")
    assert resp.status_code == 400
    assert "provider" in resp.json()["detail"]


def test_get_run_requires_auth():
    app = create_app(settings=ServerSettings(api_keys={"secret"}))
    with TestClient(app) as tc:
        resp = tc.get("/api/v1/runs/abc")
        assert resp.status_code == 401


def test_get_run_missing_is_404(tmp_path):
    app = create_app(settings=ServerSettings(no_auth=True, db_path=tmp_path / "empty.db"))
    with TestClient(app) as tc:
        resp = tc.get("/api/v1/runs/no-such-run")
    assert resp.status_code == 404


def test_get_run_returns_sanitized_detail(tmp_path):
    from atomics.storage.records import EvaluationResultRecord
    from atomics.storage.repository import MetricsRepository

    db = tmp_path / "runs.db"
    repo = MetricsRepository(db)
    repo.create_run("run-detail", tier="refusal", provider="ollama", model="qwen")
    repo.save_evaluation_result(
        EvaluationResultRecord(
            run_id="run-detail",
            suite="refusal",
            fixture_id="rf-01",
            status="complete",
            generation_status="completed",
            judge_status="scored",
            latency_ms=10.0,
            input_tokens=4,
            output_tokens=2,
            total_tokens=6,
            score=0.8,
            result_json={"secret": "should-not-leak"},
            provider="ollama",
            model="qwen",
        )
    )
    repo.close()

    app = create_app(settings=ServerSettings(no_auth=True, db_path=db))
    with TestClient(app) as tc:
        resp = tc.get("/api/v1/runs/run-detail")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run"]["run_id"] == "run-detail"
    assert body["fixtures"][0]["id"] == "rf-01"
    assert body["fixtures"][0]["score"] == 0.8
    assert "result_json" not in str(body)
    assert "should-not-leak" not in str(body)


def test_trends_requires_auth():
    app = create_app(settings=ServerSettings(api_keys={"secret"}))
    with TestClient(app) as tc:
        resp = tc.get("/api/v1/reports/trends")
        assert resp.status_code == 401


def test_trends_empty(tmp_path):
    app = create_app(
        settings=ServerSettings(no_auth=True, db_path=tmp_path / "empty-trends.db")
    )
    with TestClient(app) as tc:
        resp = tc.get("/api/v1/reports/trends")
    assert resp.status_code == 200
    assert resp.json() == {"hours": 24, "rows": []}


@pytest.mark.parametrize("hours", [0, 169, -1])
def test_trends_rejects_hours_outside_1_to_168(tmp_path, hours):
    app = create_app(
        settings=ServerSettings(no_auth=True, db_path=tmp_path / "bound.db")
    )
    with TestClient(app) as tc:
        resp = tc.get(f"/api/v1/reports/trends?hours={hours}")
    assert resp.status_code == 422


def test_trends_includes_eval_tokens_and_omits_prompts(tmp_path):
    from atomics.storage.records import EvaluationResultRecord
    from atomics.storage.repository import MetricsRepository

    db = tmp_path / "trends.db"
    repo = MetricsRepository(db)
    repo.create_run("trend-api", tier="refusal", provider="ollama", model="qwen")
    repo.save_evaluation_result(
        EvaluationResultRecord(
            run_id="trend-api",
            suite="refusal",
            fixture_id="rf-01",
            status="complete",
            generation_status="completed",
            judge_status="scored",
            latency_ms=10.0,
            input_tokens=4,
            output_tokens=6,
            total_tokens=10,
            estimated_cost_usd=0.03,
            result_json={"prompt": "trend-secret-prompt"},
            provider="ollama",
            model="qwen",
        )
    )
    repo.close()

    app = create_app(settings=ServerSettings(no_auth=True, db_path=db))
    with TestClient(app) as tc:
        resp = tc.get("/api/v1/reports/trends?hours=24")
    assert resp.status_code == 200
    body = resp.json()
    assert body["hours"] == 24
    assert len(body["rows"]) == 1
    assert body["rows"][0]["total_tokens"] == 10
    assert body["rows"][0]["task_count"] == 1
    assert "trend-secret-prompt" not in resp.text
    assert "prompt" not in body["rows"][0]

