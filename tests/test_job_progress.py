"""Resolved job request, truncated fixture rows, live reporter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atomics.api.job_progress import (
    RESPONSE_LIMIT,
    EvalJobReporter,
    LoadJobReporter,
    SweepJobReporter,
    eval_fixture_total,
    fixture_row,
    resolve_eval_request,
    resolve_inference_host,
    short_request,
    soak_job_total,
    stress_job_total,
    sweep_job_total,
    truncate_response,
)
from atomics.api.jobs import Job, JobStatus
from atomics.api.models import EvalRequest, SoakRequest, StressRequest, SweepRequest
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
    assert request["runs"] == 1
    assert "judge_host" not in request


def test_resolve_eval_request_keeps_runs_and_judge_host() -> None:
    settings = AtomicsSettings()
    payload = EvalRequest(
        suite="toolcall",
        provider="ollama",
        model="m",
        host="http://192.168.1.239:11434",
        judge_host="http://192.168.1.79:11434",
        runs=3,
    )
    request = resolve_eval_request(payload, settings)
    assert request["runs"] == 3
    assert request["host"] == "http://192.168.1.239:11434"
    assert request["judge_host"] == "http://192.168.1.79:11434"


def test_short_request_keeps_judge_host() -> None:
    assert short_request(
        {"suite": "toolcall", "model": "m", "host": "h", "judge_host": "j"}
    ) == {"suite": "toolcall", "model": "m", "host": "h", "judge_host": "j"}
    assert short_request(
        {"suite": "accuracy", "provider": "ollama", "model": "m", "host": "h", "budget_usd": 1}
    ) == {"suite": "accuracy", "model": "m", "host": "h"}


def test_short_request_keeps_sweep_models_and_suites() -> None:
    assert short_request(
        {
            "provider": "ollama",
            "models": ["a", "b"],
            "suites": ["eval", "refusal"],
            "budget_usd": 2,
        }
    ) == {"models": ["a", "b"], "suites": ["eval", "refusal"]}


def test_fixture_row_reads_refusal_shaped_result() -> None:
    fr = SimpleNamespace(
        fixture=SimpleNamespace(id="rf-01"),
        score=1.0,
        response_text="I will not help with that.",
        error=None,
        latency_ms=12.0,
        estimated_cost_usd=0.0,
        attempts=[SimpleNamespace(total_tokens=40)],
        task_result=None,
        judge=None,
    )
    row = fixture_row(fr)
    assert row["id"] == "rf-01"
    assert row["score"] == 1.0
    assert row["tokens"] == 40
    assert "will not help" in (row["response"] or "")


def test_fixture_row_reads_toolcall_dict() -> None:
    row = fixture_row(
        {
            "id": "tc-01",
            "tool_outcome": "dangerous_call",
            "response": "Here is the hash file.",
            "latency_ms": 9,
            "total_tokens": 20,
        }
    )
    assert row["id"] == "tc-01"
    assert row["score"] == 0.0
    assert row["tokens"] == 20


def test_eval_fixture_total_is_per_suite() -> None:
    toolcall = EvalRequest(suite="toolcall", provider="ollama")
    accuracy = EvalRequest(suite="accuracy", provider="ollama")
    assert eval_fixture_total(toolcall) == 20
    assert eval_fixture_total(accuracy) == 25


def test_eval_fixture_total_matches_suite_catalogs() -> None:
    from atomics.eval.codereview.fixtures import SECURE_CODE_FIXTURES
    from atomics.eval.refusal.fixtures import REFUSAL_FIXTURES
    from atomics.eval.toolcall.fixtures import ALL_FIXTURES as TOOLCALL_FIXTURES

    assert eval_fixture_total(EvalRequest(suite="toolcall", provider="ollama")) == len(
        TOOLCALL_FIXTURES
    )
    assert eval_fixture_total(EvalRequest(suite="refusal", provider="ollama")) == len(
        REFUSAL_FIXTURES
    )
    assert eval_fixture_total(EvalRequest(suite="codereview", provider="ollama")) == len(
        SECURE_CODE_FIXTURES
    )


def test_fixture_row_reads_codegen_pass_rate() -> None:
    tr = TaskResult(
        run_id="r",
        category=TaskCategory.GENERAL_QA,
        task_name="cg-01",
        provider="ollama",
        model="m",
    )
    tr.status = TaskStatus.SUCCESS
    tr.response = "def add(a, b): return a + b"
    tr.total_tokens = 11
    tr.latency_ms = 4.0
    row = fixture_row(
        SimpleNamespace(
            fixture=SimpleNamespace(id="cg-01"),
            task_result=tr,
            judge=None,
            pass_rate=1.0,
        )
    )
    assert row["id"] == "cg-01"
    assert row["score"] == 1.0
    assert row["tokens"] == 11


def test_fixture_row_reads_adversarial_resistance() -> None:
    row = fixture_row(
        SimpleNamespace(
            fixture=SimpleNamespace(id="adv-01"),
            response="I will not jailbreak.",
            resistance=SimpleNamespace(score=0.8),
            latency_ms=15.0,
            error=None,
            attempts=[SimpleNamespace(total_tokens=33)],
            task_result=None,
            judge=None,
        )
    )
    assert row["id"] == "adv-01"
    assert row["score"] == 0.8
    assert row["tokens"] == 33


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


def test_stress_job_total_is_concurrency_ladder() -> None:
    payload = StressRequest(
        provider="ollama",
        model="m",
        budget_usd=1.0,
        max_concurrency=2,
        phase_seconds=5.0,
    )
    assert stress_job_total(payload) == 2


def test_soak_job_total_is_duration_over_interval() -> None:
    payload = SoakRequest(
        provider="ollama",
        model="m",
        budget_usd=1.0,
        duration_seconds=30,
        sample_interval=10,
    )
    assert soak_job_total(payload) == 2


def test_load_reporter_grows_phases_and_clears_in_flight() -> None:
    job = Job(job_id="l", kind="stress", status=JobStatus.RUNNING, created_at=0.0)
    reporter = LoadJobReporter(
        job,
        kind="stress",
        meta={"provider": "ollama", "model": "m"},
        rows_key="phases",
        total=2,
    )
    reporter.start({"concurrency": 1, "phase_seconds": 5.0})
    assert job.progress == {
        "current": 0,
        "total": 2,
        "in_flight": {"concurrency": 1, "phase_seconds": 5.0},
        "trail": [{"concurrency": 1, "phase_seconds": 5.0}],
    }
    reporter.done(
        {
            "concurrency": 1,
            "requests": 7,
            "failed": 0,
            "aggregate_tps": 337.0,
            "avg_latency_ms": 725.0,
            "p95_latency_ms": 733.0,
        }
    )
    assert job.progress["current"] == 1
    assert job.progress["in_flight"] is None
    assert job.progress["trail"] == [{"concurrency": 1, "phase_seconds": 5.0}]
    assert job.result is not None
    assert job.result["phases"][0]["concurrency"] == 1


def test_sweep_job_total_is_models_times_suites() -> None:
    payload = SweepRequest(
        provider="ollama",
        models=["a", "b"],
        suites=["eval", "refusal"],
        budget_usd=1.0,
    )
    assert sweep_job_total(payload) == 4


def test_sweep_reporter_grows_jobs_and_clears_in_flight() -> None:
    job = Job(job_id="s", kind="sweep", status=JobStatus.RUNNING, created_at=0.0)
    reporter = SweepJobReporter(
        job,
        provider="ollama",
        models=["a", "b"],
        suites=["eval"],
        runs=1,
        budget_usd=2.0,
        total=2,
    )
    reporter.start("a", "eval")
    assert job.progress == {
        "current": 0,
        "total": 2,
        "in_flight": {"model": "a", "suite": "eval"},
        "trail": [{"model": "a", "suite": "eval"}],
    }
    reporter.done(
        SimpleNamespace(
            model="a",
            suite="eval",
            ok=True,
            headline=0.9,
            error=None,
            tool_capable=None,
            exit_code=0,
        )
    )
    assert job.progress["current"] == 1
    assert job.progress["in_flight"] is None
    assert job.progress["trail"] == [{"model": "a", "suite": "eval"}]
    assert job.result is not None
    assert job.result["jobs"][0]["model"] == "a"
    assert job.result["jobs"][0]["headline"] == 0.9
    assert job.result["ok"] == 1


def test_sweep_reporter_appends_start_to_trail() -> None:
    job = Job(job_id="s", kind="sweep", status=JobStatus.RUNNING, created_at=0.0)
    reporter = SweepJobReporter(
        job,
        provider="ollama",
        models=["a", "b"],
        suites=["eval"],
        runs=1,
        budget_usd=2.0,
        total=2,
    )
    reporter.start("a", "eval")
    reporter.start("b", "eval")
    assert job.progress["trail"] == [
        {"model": "a", "suite": "eval"},
        {"model": "b", "suite": "eval"},
    ]


def test_sweep_trail_caps_at_total() -> None:
    job = Job(job_id="s", kind="sweep", status=JobStatus.RUNNING, created_at=0.0)
    reporter = SweepJobReporter(
        job,
        provider="ollama",
        models=["a", "b"],
        suites=["eval"],
        runs=1,
        budget_usd=2.0,
        total=2,
    )
    reporter.start("a", "eval")
    reporter.start("b", "eval")
    reporter.start("c", "eval")
    assert job.progress["trail"] == [
        {"model": "a", "suite": "eval"},
        {"model": "b", "suite": "eval"},
    ]


def test_eval_trail_keeps_phases_past_twice_total() -> None:
    # Multiturn: generate + per-turn judge + conversation judge on one fixture.
    job = Job(job_id="j", kind="eval", status=JobStatus.RUNNING, created_at=0.0)
    reporter = EvalJobReporter(
        job,
        suite="multiturn",
        provider="ollama",
        model="m",
        judge_model="m",
        host=None,
        total=1,
    )
    reporter.phase("mt-01", "generate", "m")
    reporter.phase("mt-01", "judge", "m")
    reporter.phase("mt-01", "judge", "m")
    assert [row["phase"] for row in job.progress["trail"]] == [
        "generate",
        "judge",
        "judge",
    ]


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
    assert job.progress == {"current": 0, "total": 2, "in_flight": None, "trail": []}
    assert job.result is None
    reporter.phase("ev-01", "generate", "llama3.2:1b")
    reporter.phase("ev-01", "judge", "llama3.2:1b")
    assert job.progress["in_flight"] == {
        "fixture_id": "ev-01",
        "phase": "judge",
        "model": "llama3.2:1b",
    }
    assert job.progress["trail"] == [
        {"fixture_id": "ev-01", "phase": "generate", "model": "llama3.2:1b"},
        {"fixture_id": "ev-01", "phase": "judge", "model": "llama3.2:1b"},
    ]
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
    assert job.progress["trail"] == [
        {"fixture_id": "ev-01", "phase": "generate", "model": "llama3.2:1b"},
        {"fixture_id": "ev-01", "phase": "judge", "model": "llama3.2:1b"},
    ]
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


@pytest.mark.asyncio
async def test_run_rag_on_phase_generate_then_judge() -> None:
    from atomics.eval.rag.fixtures import ALL_RAG_FIXTURES
    from atomics.eval.rag.runner import run_rag

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
    scored = SimpleNamespace(
        score=0.8,
        rationale="ok",
        parse_failed=False,
        grounding=3,
        faithfulness=3,
        abstention=2,
    )

    with patch("atomics.eval.rag.runner.score_rag_consensus", new=AsyncMock(return_value=scored)):
        await run_rag(
            provider,
            judge_provider=judge,
            model="under-test",
            judge_model="judge-tag",
            fixtures=[ALL_RAG_FIXTURES[0]],
            on_phase=lambda fid, phase, model: phases.append((fid, phase, model)),
        )
    assert phases[0] == (ALL_RAG_FIXTURES[0].id, "generate", "under-test")
    assert phases[1] == (ALL_RAG_FIXTURES[0].id, "judge", "judge-tag")


@pytest.mark.asyncio
async def test_run_codegen_on_phase_generate_only() -> None:
    from atomics.eval.codegen.fixtures import ALL_CODEGEN_FIXTURES
    from atomics.eval.codegen.runner import run_codegen

    phases: list[str] = []
    provider = MagicMock()
    provider.name = "test"
    provider.generate = AsyncMock(side_effect=ConnectionError("down"))
    provider.default_model = "m"

    await run_codegen(
        provider,
        model="m",
        fixtures=[ALL_CODEGEN_FIXTURES[0]],
        on_phase=lambda _fid, phase, _model: phases.append(phase),
    )
    assert phases == ["generate"]
