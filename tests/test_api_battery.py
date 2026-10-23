"""Tests for the API battery runner."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from atomics.api._battery import run_battery_from_request
from atomics.api.models import BatteryRequest


def _stub_live_steps(monkeypatch) -> None:
    """Desk-pass probes the provider and runs a QA file; keep both offline."""
    from atomics.benchmark.qa_runner import QASuiteResult

    async def fake_provider_test(payload):
        return {"ok": True, "health": True}

    async def fake_run_qa_suite(**kwargs):
        return QASuiteResult(model="x", host="http://x")

    monkeypatch.setattr("atomics.api._battery.run_provider_test", fake_provider_test)
    monkeypatch.setattr("atomics.api._battery.load_qa_suite", lambda _path: ("", "http://x", []))
    monkeypatch.setattr("atomics.api._battery.run_qa_suite", fake_run_qa_suite)


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
    _stub_live_steps(monkeypatch)
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


def _capture_eval_steps(monkeypatch) -> dict:
    captured: dict = {}

    async def fake_run_eval_suite(payload, job=None):
        captured[payload.suite] = payload
        return {"suite": payload.suite, "fixtures": []}

    monkeypatch.setattr("atomics.api._battery.run_eval_suite", fake_run_eval_suite)
    return captured


@pytest.mark.asyncio
async def test_battery_steps_run_the_cli_subset(monkeypatch):
    from atomics.eval.adversarial import select_fixtures as adversarial_fixtures
    from atomics.eval.redblue.fixtures import BLUE_FIXTURES
    from atomics.eval.toolcall.fixtures import fixtures_for_category

    captured = _capture_eval_steps(monkeypatch)
    await run_battery_from_request(
        BatteryRequest(name="agent-gate", provider="ollama", model="x", budget_usd=5)
    )
    wanted = adversarial_fixtures(("mcp", "tool_safety", "tool_desc_injection"))
    assert captured["adversarial"].fixtures == [f.id for f in wanted]
    assert captured["toolcall"].fixtures == [f.id for f in fixtures_for_category("direct")]

    await run_battery_from_request(
        BatteryRequest(name="blue-capability", provider="ollama", model="x", budget_usd=5)
    )
    assert captured["redblue"].fixtures == [f.id for f in BLUE_FIXTURES]
    assert captured["refusal"].fixtures == [
        "rc-b01",
        "rc-b02",
        "rc-b03",
        "rc-b04",
        "rc-b05",
        "rc-b06",
    ]
    assert captured["codereview"].fixtures is None


@pytest.mark.asyncio
async def test_battery_runs_default_per_step_and_reach_every_judged_suite(monkeypatch):
    captured = _capture_eval_steps(monkeypatch)
    gate = {"name": "agent-gate", "provider": "ollama", "model": "x", "budget_usd": 5}

    await run_battery_from_request(BatteryRequest(**gate))
    assert (captured["adversarial"].runs, captured["toolcall"].runs) == (1, 3)

    await run_battery_from_request(BatteryRequest(**gate, runs=2))
    assert (captured["adversarial"].runs, captured["toolcall"].runs) == (2, 2)

    result = await run_battery_from_request(
        BatteryRequest(name="blue-capability", provider="ollama", model="x", budget_usd=5, runs=3)
    )
    assert [s["suite"] for s in result["steps"]] == ["redblue", "codereview", "refusal"]
    assert [captured[s].runs for s in ("redblue", "codereview", "refusal")] == [3, 3, 3]


@pytest.mark.asyncio
async def test_battery_pins_temperature_on_every_eval_step(monkeypatch):
    captured = _capture_eval_steps(monkeypatch)
    await run_battery_from_request(
        BatteryRequest(
            name="agent-gate", provider="ollama", model="x", budget_usd=5, temperature=0.0
        )
    )
    assert {suite: p.temperature for suite, p in captured.items()} == {
        "adversarial": 0.0,
        "toolcall": 0.0,
    }


@pytest.mark.asyncio
async def test_battery_step_rows_stay_with_their_step(monkeypatch):
    from atomics.api.jobs import Job, JobStatus

    async def reporting_run_eval_suite(payload, job=None):
        # What EvalJobReporter does: live rows accumulate on the job.
        if job.result is None:
            job.result = {"fixtures": []}
        job.result["fixtures"].append({"id": payload.suite})
        return {"suite": payload.suite, "fixtures": list(job.result["fixtures"])}

    monkeypatch.setattr("atomics.api._battery.run_eval_suite", reporting_run_eval_suite)
    job = Job(job_id="b", kind="battery", status=JobStatus.RUNNING, created_at=0.0)
    result = await run_battery_from_request(
        BatteryRequest(name="agent-gate", provider="ollama", model="x", budget_usd=5), job=job
    )
    assert [s["result"]["fixtures"] for s in result["steps"]] == [
        [{"id": "adversarial"}],
        [{"id": "toolcall"}],
    ]


@pytest.mark.asyncio
async def test_battery_defaults_to_no_thinking_like_the_cli(monkeypatch):
    captured = _capture_eval_steps(monkeypatch)
    await run_battery_from_request(
        BatteryRequest(name="agent-gate", provider="ollama", model="x", budget_usd=5)
    )
    assert {p.thinking for p in captured.values()} == {False}


@pytest.mark.asyncio
async def test_battery_pins_temperature_on_raw_qa(monkeypatch):
    seen: dict = {}

    async def fake_run_qa_suite(**kwargs):
        from atomics.benchmark.qa_runner import QASuiteResult

        seen.update(kwargs)
        return QASuiteResult(model="x", host="http://x")

    _capture_eval_steps(monkeypatch)
    monkeypatch.setattr(
        "atomics.api._battery.run_provider_test", AsyncMock(return_value={"ok": True})
    )
    monkeypatch.setattr("atomics.api._battery.load_qa_suite", lambda path: ("", "http://x", []))
    monkeypatch.setattr("atomics.api._battery.run_qa_suite", fake_run_qa_suite)
    await run_battery_from_request(
        BatteryRequest(
            name="desk-pass", provider="ollama", model="x", budget_usd=5, temperature=0.3
        )
    )
    assert seen["temperature"] == 0.3


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
    _stub_live_steps(monkeypatch)

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
        TestClient(app, base_url="http://127.0.0.1") as client,
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
    with TestClient(app, base_url="http://127.0.0.1") as client:
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
    with TestClient(app, base_url="http://127.0.0.1") as client:
        resp = client.post(
            "/api/v1/batteries",
            json={"name": "desk-pass", "provider": "ollama", "model": "x"},
        )
    assert resp.status_code == 422


@pytest.mark.parametrize("path", ["/etc/hosts", "../secrets.yaml", "profiles/../.env", "qa/x.yaml"])
def test_battery_profile_must_live_under_profiles(path):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="profiles/"):
        BatteryRequest(name="desk-pass", provider="ollama", model="x", budget_usd=5, profile=path)


def test_battery_profile_under_profiles_is_accepted():
    req = BatteryRequest(
        name="desk-pass",
        provider="ollama",
        model="x",
        budget_usd=5,
        profile="profiles/examples/a.yaml",
    )
    assert req.profile == "profiles/examples/a.yaml"
