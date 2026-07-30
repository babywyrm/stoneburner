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
    assert "<script>" in res.text
    assert "Recent runs" in res.text
    assert "Distributed jobs" in res.text
    assert "Workers" in res.text
    assert "Compare by provider" in res.text


def test_dashboard_lists_no_data_when_empty(client_with_dashboard):
    res = client_with_dashboard.get("/api/v1/distributed/runs")
    assert res.status_code == 200
    assert res.json() == {"jobs": []}


@pytest.mark.parametrize(
    "path",
    ["/api/v1/distributed/runs", "/api/v1/workers"],
)
def test_dashboard_data_endpoints_require_auth_when_not_no_auth(path):
    app = create_app(ServerSettings(no_auth=False, api_keys={"secret"}, with_dashboard=True))
    with TestClient(app) as tc:
        res = tc.get(path)
        assert res.status_code == 401

        res = tc.get(path, headers={"X-API-Key": "secret"})
        assert res.status_code == 200
