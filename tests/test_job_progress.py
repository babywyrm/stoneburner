"""Resolved job request, truncated fixture rows, live reporter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from atomics.api.job_progress import (
    RESPONSE_LIMIT,
    EvalJobReporter,
    eval_fixture_total,
    fixture_row,
    resolve_eval_request,
    resolve_inference_host,
    short_request,
    truncate_response,
)
from atomics.api.jobs import Job, JobStatus
from atomics.api.models import EvalRequest
from atomics.config import AtomicsSettings
from atomics.eval.fixtures import EVAL_FIXTURES
from atomics.eval.runner import run_eval
from atomics.models import TaskCategory, TaskResult, TaskStatus
from atomics.providers.base import ProviderResponse


def test_truncate_response_caps_at_500() -> None:
    assert truncate_response("x" * 500) == "x" * 500
    assert truncate_response("x" * 501) == "x" * 500
    assert truncate_response(None) is None


def test_resolve_inference_host_prefers_payload() -> None:
    settings = AtomicsSettings(ollama_host="http://localhost:11434")
    assert (
        resolve_inference_host("ollama", "http://192.168.1.79:11434", settings)
        == "http://192.168.1.79:11434"
    )
    assert resolve_inference_host("ollama", None, settings) == "http://localhost:11434"


def test_resolve_eval_request_fills_default_judge_and_host() -> None:
    settings = AtomicsSettings()
    payload = EvalRequest(suite="accuracy", provider="ollama", model="llama3.2:1b")
    request = resolve_eval_request(payload, settings)
    assert request["suite"] == "accuracy"
    assert request["provider"] == "ollama"
    assert request["model"] == "llama3.2:1b"
    assert request["judge_model"] == settings.ollama_model
    assert request["host"] == settings.ollama_host


def test_short_request_keeps_suite_model_host() -> None:
    assert short_request(
        {"suite": "accuracy", "provider": "ollama", "model": "m", "host": "h", "budget_usd": 1}
    ) == {"suite": "accuracy", "model": "m", "host": "h"}


def test_eval_fixture_total_honours_ids() -> None:
    assert eval_fixture_total(EvalRequest(suite="accuracy", provider="ollama")) == len(
        EVAL_FIXTURES
    )
    first = EVAL_FIXTURES[0].id
    payload = EvalRequest(suite="accuracy", provider="ollama", fixtures=[first])
    assert eval_fixture_total(payload) == 1


def test_fixture_row_failed_has_error_and_no_score() -> None:
    fixture = EVAL_FIXTURES[0]
    tr = TaskResult(
        run_id="r",
        category=TaskCategory.GENERAL_QA,
        task_name=fixture.id,
        provider="ollama",
        model="m",
    )
    tr.status = TaskStatus.FAILED
    tr.error_message = "ConnectError: down"
    fr = SimpleNamespace(fixture=fixture, task_result=tr, judge=None)
    row = fixture_row(fr)
    assert row["id"] == fixture.id
    assert row["status"] == "failed"
    assert row["score"] is None
    assert row["error"] == "ConnectError: down"
    assert row["response"] is None


def test_fixture_row_truncates_response() -> None:
    fixture = EVAL_FIXTURES[0]
    tr = TaskResult(
        run_id="r",
        category=TaskCategory.GENERAL_QA,
        task_name=fixture.id,
        provider="ollama",
        model="m",
    )
    tr.status = TaskStatus.SUCCESS
    tr.response = "z" * (RESPONSE_LIMIT + 20)
    tr.total_tokens = 9
    tr.latency_ms = 12.34
    fr = SimpleNamespace(
        fixture=fixture,
        task_result=tr,
        judge=SimpleNamespace(score=0.8, parse_failed=False),
    )
    row = fixture_row(fr)
    assert row["status"] == "success"
    assert row["score"] == 0.8
    assert row["tokens"] == 9
    assert row["response"] == "z" * RESPONSE_LIMIT


def test_reporter_grows_result_and_clears_in_flight() -> None:
    job = Job(job_id="j", kind="eval", status=JobStatus.RUNNING, created_at=0.0)
    reporter = EvalJobReporter(
        job,
        suite="accuracy",
        provider="ollama",
        model="llama3.2:1b",
        judge_model="llama3.2:1b",
        host="http://192.168.1.79:11434",
        total=2,
    )
    assert job.progress == {"current": 0, "total": 2, "in_flight": None}
    assert job.result is None
    reporter.phase("ev-01", "generate", "llama3.2:1b")
    assert job.progress["in_flight"] == {
        "fixture_id": "ev-01",
        "phase": "generate",
        "model": "llama3.2:1b",
    }
    fixture = EVAL_FIXTURES[0]
    tr = TaskResult(
        run_id="r",
        category=TaskCategory.GENERAL_QA,
        task_name=fixture.id,
        provider="ollama",
        model="m",
    )
    tr.status = TaskStatus.SUCCESS
    tr.response = "4"
    tr.total_tokens = 46
    tr.latency_ms = 2100
    tr.estimated_cost_usd = 0.0
    reporter.fixture_done(
        SimpleNamespace(
            fixture=fixture,
            task_result=tr,
            judge=SimpleNamespace(score=0.8, parse_failed=False),
        )
    )
    assert job.progress["current"] == 1
    assert job.progress["in_flight"] is None
    assert job.result is not None
    assert job.result["fixtures_run"] == 1
    assert job.result["fixtures"][0]["id"] == fixture.id
    assert job.result["host"] == "http://192.168.1.79:11434"


@pytest.mark.asyncio
async def test_run_eval_on_phase_generate_then_judge() -> None:
    phases: list[tuple[str, str, str | None]] = []
    resp = ProviderResponse(
        text="ok",
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        model="m",
        latency_ms=1.0,
        estimated_cost_usd=0.0,
    )
    provider = MagicMock()
    provider.name = "test"
    provider.generate = AsyncMock(return_value=resp)
    provider.default_model = "under-test"
    judge = MagicMock()
    judge.name = "ollama"
    judge.default_model = "judge-tag"
    judge.generate = AsyncMock(
        return_value=ProviderResponse(
            text="ACCURACY: 4\nCOMPLETENESS: 3\nFORMAT: 3\nRATIONALE: Good.",
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            model="judge-tag",
            latency_ms=1.0,
            estimated_cost_usd=0.0,
        )
    )

    def on_phase(fixture_id: str, phase: str, model: str | None) -> None:
        phases.append((fixture_id, phase, model))

    await run_eval(
        provider,
        judge_provider=judge,
        model="under-test",
        judge_model="judge-tag",
        fixtures=[EVAL_FIXTURES[0]],
        on_phase=on_phase,
    )
    assert phases[0] == (EVAL_FIXTURES[0].id, "generate", "under-test")
    assert phases[1] == (EVAL_FIXTURES[0].id, "judge", "judge-tag")


@pytest.mark.asyncio
async def test_run_eval_skips_judge_phase_on_generate_failure() -> None:
    phases: list[str] = []
    provider = MagicMock()
    provider.name = "failing"
    provider.generate = AsyncMock(side_effect=ConnectionError("down"))
    provider.default_model = "m"
    judge = MagicMock()
    judge.name = "ollama"
    judge.default_model = "j"
    judge.generate = AsyncMock(side_effect=AssertionError("judge must not run"))

    await run_eval(
        provider,
        judge_provider=judge,
        fixtures=[EVAL_FIXTURES[0]],
        on_phase=lambda _fid, phase, _model: phases.append(phase),
    )
    assert phases == ["generate"]
