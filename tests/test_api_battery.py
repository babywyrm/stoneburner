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
