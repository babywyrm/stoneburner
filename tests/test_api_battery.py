"""Tests for the API battery runner."""

from __future__ import annotations

import pytest

from atomics.api._battery import run_battery_from_request
from atomics.api.models import BatteryRequest


@pytest.mark.asyncio
async def test_battery_expands_steps_in_order(monkeypatch):
    seen: list[str] = []

    async def fake_run_eval_suite(payload, job=None):
        seen.append(payload.suite)
        return {"suite": payload.suite, "fixtures": []}

    monkeypatch.setattr("atomics.api._battery.run_eval_suite", fake_run_eval_suite)
    req = BatteryRequest(name="desk-pass", provider="ollama", model="x", budget_usd=5)
    result = await run_battery_from_request(req)
    assert [s["suite"] for s in result["steps"] if not s.get("skipped")] == seen
    assert seen  # desk-pass has eval-suite steps
    assert result["battery"] == "desk-pass"


@pytest.mark.asyncio
async def test_battery_forwards_thinking_and_effort(monkeypatch):
    captured: list = []

    async def fake_run_eval_suite(payload, job=None):
        captured.append(payload)
        return {"suite": payload.suite, "fixtures": []}

    monkeypatch.setattr("atomics.api._battery.run_eval_suite", fake_run_eval_suite)
    req = BatteryRequest(
        name="desk-pass",
        provider="ollama",
        model="x",
        budget_usd=5,
        thinking=True,
        effort="low",
    )
    await run_battery_from_request(req)
    assert captured
    for payload in captured:
        assert payload.thinking is True
        assert payload.effort == "low"


@pytest.mark.asyncio
async def test_battery_skips_provider_test_and_qa(monkeypatch):
    seen: list[str] = []

    async def fake_run_eval_suite(payload, job=None):
        seen.append(payload.suite)
        return {"suite": payload.suite, "fixtures": []}

    monkeypatch.setattr("atomics.api._battery.run_eval_suite", fake_run_eval_suite)
    req = BatteryRequest(name="desk-pass", provider="ollama", model="x", budget_usd=5)
    result = await run_battery_from_request(req)
    assert "provider-test" not in seen
    assert "qa" not in seen
    skipped = {s["suite"] for s in result["steps"] if s.get("skipped")}
    assert "provider-test" in skipped
    assert "qa" in skipped


def test_post_batteries_returns_202_and_kind():
    from unittest.mock import AsyncMock, patch

    from fastapi.testclient import TestClient

    from atomics.api.config import ServerSettings
    from atomics.api.server import create_app

    app = create_app(settings=ServerSettings(no_auth=True))
    with (
        patch(
            "atomics.api.routes.run_battery_from_request",
            new_callable=AsyncMock,
            return_value={"battery": "desk-pass", "steps": []},
        ),
        TestClient(app) as client,
    ):
        resp = client.post(
            "/api/v1/batteries",
            json={
                "name": "desk-pass",
                "provider": "ollama",
                "model": "x",
                "budget_usd": 5,
            },
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["kind"] == "battery"
    assert body["progress"]["total"] >= 1


def test_post_batteries_unknown_name_is_422():
    from fastapi.testclient import TestClient

    from atomics.api.config import ServerSettings
    from atomics.api.server import create_app

    app = create_app(settings=ServerSettings(no_auth=True))
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/batteries",
            json={"name": "nope", "provider": "ollama", "model": "x", "budget_usd": 5},
        )
    assert resp.status_code == 422


def test_post_batteries_without_budget_is_422():
    from fastapi.testclient import TestClient

    from atomics.api.config import ServerSettings
    from atomics.api.server import create_app

    app = create_app(settings=ServerSettings(no_auth=True))
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/batteries",
            json={"name": "desk-pass", "provider": "ollama", "model": "x"},
        )
    assert resp.status_code == 422
