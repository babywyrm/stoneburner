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

    async def fake_provider_test(payload):
        return {"ok": True, "health": True}

    def fake_load_qa_suite(path):
        return "", "http://x", []

    async def fake_run_qa_suite(**kwargs):
        from atomics.benchmark.qa_runner import QASuiteResult

        return QASuiteResult(model="x", host="http://x")

    monkeypatch.setattr("atomics.api._battery.run_eval_suite", fake_run_eval_suite)
    monkeypatch.setattr("atomics.api._battery.run_provider_test", fake_provider_test)
    monkeypatch.setattr("atomics.api._battery.load_qa_suite", fake_load_qa_suite)
    monkeypatch.setattr("atomics.api._battery.run_qa_suite", fake_run_qa_suite)
    req = BatteryRequest(name="desk-pass", provider="ollama", model="x", budget_usd=5)
    result = await run_battery_from_request(req)
    eval_steps = [s["suite"] for s in result["steps"] if s["suite"] in seen]
    assert eval_steps == seen
    assert seen  # desk-pass has eval-suite steps
    assert result["battery"] == "desk-pass"
    # provider-test and qa run before the eval steps
    assert result["steps"][0]["suite"] == "provider-test"
    assert result["steps"][1]["suite"] == "qa"


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
async def test_battery_skips_only_archreview(monkeypatch):
    seen: list[str] = []

    async def fake_run_eval_suite(payload, job=None):
        seen.append(payload.suite)
        return {"suite": payload.suite, "fixtures": []}

    async def fake_provider_test(payload):
        return {"ok": True, "health": True}

    def fake_load_qa_suite(path):
        return "", "http://x", []

    async def fake_run_qa_suite(**kwargs):
        from atomics.benchmark.qa_runner import QASuiteResult

        return QASuiteResult(model="x", host="http://x")

    monkeypatch.setattr("atomics.api._battery.run_eval_suite", fake_run_eval_suite)
    monkeypatch.setattr("atomics.api._battery.run_provider_test", fake_provider_test)
    monkeypatch.setattr("atomics.api._battery.load_qa_suite", fake_load_qa_suite)
    monkeypatch.setattr("atomics.api._battery.run_qa_suite", fake_run_qa_suite)
    req = BatteryRequest(name="threat-model", provider="ollama", model="x", budget_usd=5)
    result = await run_battery_from_request(req)
    skipped = {s["suite"] for s in result["steps"] if s.get("skipped")}
    assert skipped == {"archreview"}
    assert "provider-test" not in skipped
    assert "qa" not in skipped


@pytest.mark.asyncio
async def test_battery_runs_provider_test_step(monkeypatch):
    async def fake_provider_test(payload):
        return {"ok": True, "health": True, "provider": "ollama", "model": payload.model}

    async def fake_run_eval_suite(payload, job=None):
        return {"suite": payload.suite, "fixtures": []}

    monkeypatch.setattr("atomics.api._battery.run_provider_test", fake_provider_test)
    monkeypatch.setattr("atomics.api._battery.run_eval_suite", fake_run_eval_suite)
    req = BatteryRequest(name="desk-pass", provider="ollama", model="x", budget_usd=5)
    result = await run_battery_from_request(req)
    pt = next(s for s in result["steps"] if s["suite"] == "provider-test")
    assert pt["ok"] is True
    assert pt.get("skipped") is not True


@pytest.mark.asyncio
async def test_battery_runs_qa_step_and_marks_fail(monkeypatch):
    from atomics.benchmark.qa_runner import QAFixture, QAResult, QASuiteResult

    async def fake_provider_test(payload):
        return {"ok": True, "health": True, "provider": "ollama", "model": payload.model}

    async def fake_run_eval_suite(payload, job=None):
        return {"suite": payload.suite, "fixtures": []}

    def fake_load_qa_suite(path):
        return "", "http://x", [QAFixture(id="f1", prompt="p")]

    async def fake_run_qa_suite(**kwargs):
        suite = QASuiteResult(model="x", host="http://x")
        suite.results.append(
            QAResult(
                fixture=QAFixture(id="f1", prompt="p"),
                response="r",
                latency_ms=1.0,
                status="FAIL",
            )
        )
        return suite

    monkeypatch.setattr("atomics.api._battery.run_provider_test", fake_provider_test)
    monkeypatch.setattr("atomics.api._battery.run_eval_suite", fake_run_eval_suite)
    monkeypatch.setattr("atomics.api._battery.load_qa_suite", fake_load_qa_suite)
    monkeypatch.setattr("atomics.api._battery.run_qa_suite", fake_run_qa_suite)
    req = BatteryRequest(name="desk-pass", provider="ollama", model="x", budget_usd=5)
    result = await run_battery_from_request(req)
    qa = next(s for s in result["steps"] if s["suite"] == "qa")
    assert qa["ok"] is False
    assert qa.get("skipped") is not True
    assert qa["passed"] == 0
    assert qa["total"] == 1


@pytest.mark.asyncio
async def test_battery_eval_step_not_ok_when_coverage_is_incomplete(monkeypatch):
    async def fake_provider_test(payload):
        return {"ok": True, "health": True}

    def fake_load_qa_suite(path):
        return "", "http://x", []

    async def fake_run_qa_suite(**kwargs):
        from atomics.benchmark.qa_runner import QASuiteResult

        return QASuiteResult(model="x", host="http://x")

    async def fake_run_eval_suite(payload, job=None):
        if payload.suite == "toolcall":
            return {"suite": "toolcall", "tool_capable": False}
        return {
            "suite": payload.suite,
            "integrity": {"status": "partial", "should_exit_nonzero": True},
        }

    monkeypatch.setattr("atomics.api._battery.run_provider_test", fake_provider_test)
    monkeypatch.setattr("atomics.api._battery.load_qa_suite", fake_load_qa_suite)
    monkeypatch.setattr("atomics.api._battery.run_qa_suite", fake_run_qa_suite)
    monkeypatch.setattr("atomics.api._battery.run_eval_suite", fake_run_eval_suite)

    desk = await run_battery_from_request(
        BatteryRequest(name="desk-pass", provider="ollama", model="x", budget_usd=5)
    )
    toolcall = next(s for s in desk["steps"] if s["suite"] == "toolcall")
    assert toolcall["ok"] is False

    blue = await run_battery_from_request(
        BatteryRequest(name="blue-capability", provider="ollama", model="x", budget_usd=5)
    )
    eval_steps = [s for s in blue["steps"] if s["suite"] != "archreview"]
    assert eval_steps
    assert all(s["ok"] is False for s in eval_steps)


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
