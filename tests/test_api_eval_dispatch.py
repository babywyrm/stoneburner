import time
from unittest.mock import AsyncMock, patch

import pytest

from atomics.api._runners import SUPPORTED_EVAL_SUITES


@pytest.mark.asyncio
async def test_post_evals_accuracy(client):
    with patch("atomics.api._runners.run_eval", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = type(
            "S",
            (),
            {
                "overall_accuracy": 0.9,
                "fixtures_run": 3,
                "total_tokens": 100,
                "total_cost_usd": 0.01,
                "fixture_results": [1, 2, 3],
            },
        )()
        resp = client.post("/api/v1/evals", json={"suite": "accuracy", "provider": "ollama"})
    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_post_evals_includes_resolved_request(client):
    with patch("atomics.api._runners.run_eval", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = type(
            "S",
            (),
            {
                "overall_accuracy": 0.9,
                "fixture_results": [],
                "total_tokens": 0,
                "total_cost_usd": 0.0,
            },
        )()
        resp = client.post(
            "/api/v1/evals",
            json={
                "suite": "accuracy",
                "provider": "ollama",
                "model": "llama3.2:1b",
                "host": "http://10.0.0.9:11434",
            },
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["request"]["suite"] == "accuracy"
    assert body["request"]["model"] == "llama3.2:1b"
    assert body["request"]["host"] == "http://10.0.0.9:11434"
    assert body["request"]["judge_model"]
    assert body["progress"]["current"] == 0
    assert body["progress"]["total"] >= 1
    assert body["progress"]["trail"] == []
    assert body["progress"]["in_flight"] is None
    listed = client.get("/api/v1/jobs").json()["jobs"]
    assert listed[0]["request"]["model"] == "llama3.2:1b"
    assert "fixtures" not in (listed[0].get("result") or {})


@pytest.mark.asyncio
async def test_post_evals_rag(client):
    with patch("atomics.api._runners.run_rag", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = type(
            "S",
            (),
            {
                "overall_score": 0.8,
                "fixtures_run": 2,
                "total_tokens": 50,
                "total_cost_usd": 0.005,
                "fixture_results": [1, 2],
            },
        )()
        resp = client.post("/api/v1/evals", json={"suite": "rag", "provider": "ollama"})
    assert resp.status_code == 202


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("suite", "runner_attr", "headline_attr"),
    [
        ("refusal", "run_refusal", "calibration_score"),
        ("redblue", "run_redblue", "overall_quality"),
        ("codereview", "run_codereview", "review_score"),
        ("toolcall", "run_toolcall_suite", "dangerous_call_rate"),
    ],
)
async def test_post_evals_security_suite_is_accepted(client, suite, runner_attr, headline_attr):
    summary = type(
        "S",
        (),
        {
            headline_attr: 0.8,
            "dangerous_any_run_rate": 0.8,
            "after_tool_result": {},
            "fixture_results": [1],
            "fixtures": [1],
            "total_tokens": 10,
            "total_cost_usd": 0.001,
        },
    )()
    with patch(f"atomics.api._runners.{runner_attr}", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = summary
        resp = client.post("/api/v1/evals", json={"suite": suite, "provider": "ollama"})
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        body = None
        for _ in range(80):
            poll = client.get(f"/api/v1/jobs/{job_id}")
            body = poll.json()
            if body["status"] in {"completed", "failed"}:
                break
            time.sleep(0.05)
    assert body is not None
    assert body["status"] == "completed"
    assert body["result"]["suite"] == suite
    assert body["result"]["overall_score"] == 0.8


@pytest.mark.asyncio
async def test_completed_toolcall_job_includes_fixture_rows(client):
    summary = type(
        "S",
        (),
        {
            "dangerous_call_rate": 0.25,
            "dangerous_any_run_rate": 0.5,
            "after_tool_result": {},
            "fixtures": [
                {
                    "id": "tc-01",
                    "tool_outcome": "dangerous_call",
                    "response": "Here is the hash file.",
                    "latency_ms": 9,
                    "total_tokens": 20,
                }
            ],
            "fixture_results": [],
            "total_tokens": 20,
            "total_cost_usd": 0.001,
        },
    )()
    with patch("atomics.api._runners.run_toolcall_suite", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = summary
        resp = client.post("/api/v1/evals", json={"suite": "toolcall", "provider": "ollama"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["progress"]["total"] == 35
    job_id = body["job_id"]
    completed = None
    for _ in range(80):
        poll = client.get(f"/api/v1/jobs/{job_id}")
        completed = poll.json()
        if completed["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert completed is not None
    assert completed["status"] == "completed"
    fixtures = completed["result"]["fixtures"]
    assert fixtures[0]["id"] == "tc-01"
    assert fixtures[0]["score"] == 0.0
    assert completed["result"]["overall_score"] == 0.25
    assert completed["result"]["dangerous_any_run_rate"] == 0.5


@pytest.mark.asyncio
async def test_post_evals_unsupported_suite(client):
    resp = client.post("/api/v1/evals", json={"suite": "unknown", "provider": "ollama"})
    assert resp.status_code == 400


def test_supported_eval_suites_cover_the_security_set():
    """An agent using submit_eval should see the same set the CLI security
    family uses. Missing one here is how MCP looked complete while hiding it."""
    assert {
        "adversarial",
        "refusal",
        "redblue",
        "toolcall",
        "codereview",
    }.issubset(SUPPORTED_EVAL_SUITES)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("suite", "runner_attr", "result_attr"),
    [
        ("adversarial", "run_adversarial", "fixture_results"),
        ("refusal", "run_refusal", "fixture_results"),
        ("redblue", "run_redblue", "fixture_results"),
        ("codereview", "run_codereview", "fixture_results"),
        ("toolcall", "run_toolcall_suite", "fixtures"),
    ],
)
async def test_run_eval_suite_forwards_effort_to_security_runners(suite, runner_attr, result_attr):
    """EvalRequest already accepts effort. Dropping it here is a silent no-op
    on a billed knob — the HTTP/MCP caller gets 200 and default reasoning."""
    from unittest.mock import MagicMock

    from atomics.api._runners import run_eval_suite
    from atomics.api.models import EvalRequest

    payload = EvalRequest(
        suite=suite,
        provider="ollama",
        thinking=False,
        effort="high",
        reasoning_mode="standard",
    )
    summary = type(
        "S",
        (),
        {
            result_attr: [1],
            "fixture_results": [1],
            "fixtures": [1],
            "total_tokens": 1,
            "total_cost_usd": 0.0,
            "overall_score": 0.5,
            "dangerous_any_run_rate": 0.5,
            "after_tool_result": {},
        },
    )()
    with (
        patch("atomics.api._runners._guarded_providers", return_value=(MagicMock(), MagicMock())),
        patch(f"atomics.api._runners.{runner_attr}", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.return_value = summary
        await run_eval_suite(payload)

    kwargs = mock_run.await_args.kwargs
    assert kwargs["effort"] == "high"
    assert kwargs["reasoning_mode"] == "standard"
    assert kwargs["thinking"] is False


@pytest.mark.asyncio
async def test_run_eval_suite_forwards_effort_to_codegen():
    from unittest.mock import MagicMock

    from atomics.api._runners import run_eval_suite
    from atomics.api.models import EvalRequest

    payload = EvalRequest(
        suite="codegen",
        provider="ollama",
        thinking=False,
        effort="medium",
        reasoning_mode="standard",
    )
    summary = type(
        "S",
        (),
        {
            "fixture_results": [1],
            "total_tokens": 1,
            "total_cost_usd": 0.0,
            "overall_pass_rate": 1.0,
        },
    )()
    with (
        patch("atomics.api._runners._guarded_providers", return_value=(MagicMock(), MagicMock())),
        patch("atomics.api._runners.run_codegen", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.return_value = summary
        await run_eval_suite(payload)

    kwargs = mock_run.await_args.kwargs
    assert kwargs["effort"] == "medium"
    assert kwargs["reasoning_mode"] == "standard"
    assert kwargs["thinking"] is False


def test_eval_request_validates_category_per_suite(client):
    res_bad_suite = client.post(
        "/api/v1/evals",
        json={"suite": "accuracy", "provider": "ollama", "category": "escalation"},
    )
    assert res_bad_suite.status_code == 422
    assert "does not support category filtering" in res_bad_suite.text

    res_bad_cat = client.post(
        "/api/v1/evals",
        json={"suite": "toolcall", "provider": "ollama", "category": "nonexistent"},
    )
    assert res_bad_cat.status_code == 422
    assert "unknown toolcall category" in res_bad_cat.text

    res_disjoint = client.post(
        "/api/v1/evals",
        json={
            "suite": "toolcall",
            "provider": "ollama",
            "category": "escalation",
            "fixtures": ["tc-01"],
        },
    )
    assert res_disjoint.status_code == 422
    assert "fixture IDs outside category" in res_disjoint.text


@pytest.mark.asyncio
async def test_eval_request_category_subsets_adversarial_toolcall_and_redblue():
    from atomics.api.job_progress import (
        eval_fixture_total,
        fixtures_for_request,
        resolve_eval_request,
    )
    from atomics.api.models import EvalRequest
    from atomics.config import AtomicsSettings

    adv_req = EvalRequest(suite="adversarial", provider="ollama", category="escalation")
    assert [f.id for f in fixtures_for_request(adv_req)] == [
        "mt-esc-01",
        "mt-esc-02",
        "mt-esc-03",
        "mt-esc-04",
        "mt-esc-05",
        "mt-esc-06",
    ]
    assert eval_fixture_total(adv_req) == 6
    assert resolve_eval_request(adv_req, AtomicsSettings())["category"] == "escalation"

    tc_req = EvalRequest(suite="toolcall", provider="ollama", category="direct, escalation")
    assert tc_req.category == "direct,escalation"
    tc_ids = [f.id for f in fixtures_for_request(tc_req)]
    assert tc_ids == ["tc-01", "tc-02", "tc-03", "tc-04", "tc-32", "tc-33", "tc-34", "tc-35"]

    rb_req = EvalRequest(suite="redblue", provider="ollama", category="blue")
    rb_selected = fixtures_for_request(rb_req)
    assert len(rb_selected) > 0
    assert all(f.team == "blue" for f in rb_selected)


def test_eval_request_normalizes_suite_bounds_category_and_types_catalog_without_any():
    import inspect
    from typing import Any

    from pydantic import ValidationError

    from atomics.api.job_progress import fixtures_for_request
    from atomics.api.models import EvalRequest, category_catalog_for_suite

    with pytest.raises(ValidationError, match="runs is only valid"):
        EvalRequest(suite=" ACCURACY ", provider="ollama", runs=3)

    adv_upper = EvalRequest(suite=" ADVERSARIAL ", provider="ollama", fixtures=["adv-01"])
    assert adv_upper.suite == "adversarial"
    assert [f.id for f in fixtures_for_request(adv_upper)] == ["adv-01"]

    with pytest.raises(ValidationError):
        EvalRequest(suite="toolcall", provider="ollama", category="a" * 300)

    ann = inspect.get_annotations(category_catalog_for_suite, eval_str=True)
    assert Any not in ann["return"].__args__[0].__args__


@pytest.mark.asyncio
async def test_codegen_and_multiturn_runners_respect_explicit_empty_fixtures_list():
    from unittest.mock import MagicMock

    from atomics.eval.codegen.runner import run_codegen
    from atomics.eval.multiturn.runner import run_multiturn

    provider = MagicMock()
    provider.name = "ollama"
    provider.generate = AsyncMock()
    judge = MagicMock()
    judge.name = "ollama"
    judge.generate = AsyncMock()

    cg_summary = await run_codegen(provider, fixtures=[])
    assert cg_summary.fixture_results == []
    provider.generate.assert_not_awaited()

    mt_summary = await run_multiturn(provider, judge_provider=judge, fixtures=[])
    assert mt_summary.conversation_results == []
    provider.generate.assert_not_awaited()
