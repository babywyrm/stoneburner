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

