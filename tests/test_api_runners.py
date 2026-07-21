"""Unit tests for atomics.api._runners."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import click
import pytest
from fastapi import HTTPException

from atomics.api import _runners as runners
from atomics.api.models import EvalRequest, RunRequest


def _settings(db_path=":memory:", default_model="test-model"):
    return SimpleNamespace(db_path=db_path, default_model=default_model)


@pytest.mark.asyncio
async def test_run_benchmark_from_request_returns_summary_dict():
    payload = RunRequest(provider="ollama", model="llama3", tier="ez", iterations=2)
    summary = SimpleNamespace(
        run_id="run-123",
        total_tasks=3,
        successful_tasks=2,
        failed_tasks=1,
        total_tokens=100,
        total_cost_usd=0.01,
    )
    engine = MagicMock()
    engine.run = AsyncMock(return_value=summary)
    repo = MagicMock()
    profile = SimpleNamespace(preferred_model="preferred-model")

    with (
        patch.object(runners, "load_settings", return_value=_settings()),
        patch.object(runners, "_provider_for", return_value=MagicMock(name="provider")),
        patch("atomics.core.engine.LoopEngine", return_value=engine) as mock_engine_cls,
        patch("atomics.storage.repository.MetricsRepository", return_value=repo),
        patch("atomics.tiers.get_tier_profile", return_value=profile),
    ):
        result = await runners.run_benchmark_from_request(payload)

    assert result["run_id"] == "run-123"
    assert result["provider"] == "ollama"
    assert result["model"] == "llama3"
    assert result["tier"] == "ez"
    assert result["tasks"] == 3
    assert result["success"] == 2
    assert result["failed"] == 1
    assert result["total_tokens"] == 100
    assert result["total_cost_usd"] == 0.01
    engine.run.assert_awaited_once_with(max_iterations=2)
    repo.close.assert_called_once()
    mock_engine_cls.assert_called_once()


@pytest.mark.asyncio
async def test_run_benchmark_none_summary_raises_http_400():
    payload = RunRequest(provider="ollama", iterations=1)
    engine = MagicMock()
    engine.run = AsyncMock(return_value=None)
    repo = MagicMock()

    with (
        patch.object(runners, "load_settings", return_value=_settings()),
        patch.object(runners, "_provider_for", return_value=MagicMock()),
        patch("atomics.core.engine.LoopEngine", return_value=engine),
        patch("atomics.storage.repository.MetricsRepository", return_value=repo),
        patch(
            "atomics.tiers.get_tier_profile",
            return_value=SimpleNamespace(preferred_model=None),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await runners.run_benchmark_from_request(payload)

    assert exc_info.value.status_code == 400
    repo.close.assert_called_once()


@pytest.mark.asyncio
async def test_run_eval_from_request_accuracy_normalized():
    payload = EvalRequest(
        suite="accuracy", provider="ollama", model="m1", judge_model="j1"
    )
    summary = SimpleNamespace(
        overall_accuracy=0.9,
        fixture_results=[1, 2, 3],
        total_tokens=120,
        total_cost_usd=0.02,
    )

    with (
        patch.object(runners, "_provider_for", side_effect=[MagicMock(), MagicMock()]),
        patch.object(
            runners, "run_eval", new_callable=AsyncMock, return_value=summary
        ) as mock_run,
    ):
        result = await runners.run_eval_from_request(payload)

    mock_run.assert_awaited_once()
    assert result == {
        "provider": "ollama",
        "model": "m1",
        "judge_model": "j1",
        "overall_accuracy": 0.9,
        "fixtures_run": 3,
        "total_tokens": 120,
        "total_cost_usd": 0.02,
    }


@pytest.mark.asyncio
async def test_run_eval_suite_accuracy_dispatches_to_run_eval():
    payload = EvalRequest(suite="accuracy", provider="ollama", model="m1")
    summary = SimpleNamespace(
        overall_accuracy=0.85,
        fixture_results=[1, 2],
        total_tokens=10,
        total_cost_usd=0.001,
    )

    with (
        patch.object(runners, "_provider_for", side_effect=[MagicMock(), MagicMock()]),
        patch.object(
            runners, "run_eval", new_callable=AsyncMock, return_value=summary
        ) as mock_run,
    ):
        result = await runners.run_eval_suite(payload)

    mock_run.assert_awaited_once()
    assert result["suite"] == "accuracy"
    assert result["overall_accuracy"] == 0.85
    assert result["fixtures_run"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("suite", "runner_attr", "summary", "fixtures_attr", "expected_score"),
    [
        (
            "rag",
            "run_rag",
            SimpleNamespace(
                overall_rag_score=0.8,
                fixture_results=[1, 2],
                total_tokens=50,
                total_cost_usd=0.005,
            ),
            "fixture_results",
            0.8,
        ),
        (
            "multiturn",
            "run_multiturn",
            SimpleNamespace(
                avg_conversation_score=0.7,
                conversation_results=[1],
                total_tokens=40,
                total_cost_usd=0.004,
            ),
            "conversation_results",
            0.7,
        ),
        (
            "adversarial",
            "run_adversarial",
            SimpleNamespace(
                overall_resilience=0.6,
                fixture_results=[1, 2, 3],
                total_tokens=30,
                total_cost_usd=0.003,
            ),
            "fixture_results",
            0.6,
        ),
        (
            "codegen",
            "run_codegen",
            SimpleNamespace(
                overall_pass_rate=0.5,
                fixture_results=[1],
                total_tokens=20,
                total_cost_usd=0.002,
            ),
            "fixture_results",
            0.5,
        ),
    ],
)
async def test_run_eval_suite_dispatches(
    suite, runner_attr, summary, fixtures_attr, expected_score
):
    payload = EvalRequest(
        suite=suite, provider="ollama", model="m1", judge_model="j1"
    )
    provider = MagicMock(name="provider")
    judge = MagicMock(name="judge")

    with (
        patch.object(runners, "_provider_for", side_effect=[provider, judge]),
        patch.object(
            runners, runner_attr, new_callable=AsyncMock, return_value=summary
        ) as mock_run,
    ):
        result = await runners.run_eval_suite(payload)

    mock_run.assert_awaited_once()
    assert mock_run.await_args.args[0] is provider
    if suite != "codegen":
        assert mock_run.await_args.kwargs["judge_provider"] is judge
        assert mock_run.await_args.kwargs["model"] == "m1"
        assert mock_run.await_args.kwargs["judge_model"] == "j1"
    else:
        assert mock_run.await_args.kwargs["model"] == "m1"

    assert result["suite"] == suite
    assert result["provider"] == "ollama"
    assert result["model"] == "m1"
    assert result["overall_score"] == expected_score
    assert result["fixtures_run"] == len(getattr(summary, fixtures_attr))
    assert result["total_tokens"] == summary.total_tokens
    assert result["total_cost_usd"] == summary.total_cost_usd


@pytest.mark.asyncio
async def test_run_eval_suite_unsupported_raises_value_error():
    payload = EvalRequest(suite="unknown", provider="ollama")
    with pytest.raises(ValueError, match="Unsupported eval suite"):
        await runners.run_eval_suite(payload)


def test_provider_for_click_exception_maps_to_http_400():
    with (
        patch.object(runners, "load_settings", return_value=_settings()),
        patch.object(
            runners,
            "_make_provider",
            side_effect=click.ClickException("bad provider"),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            runners._provider_for("bad", None)

    assert exc_info.value.status_code == 400
    assert "bad provider" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_run_eval_suite_propagates_provider_http_400():
    payload = EvalRequest(suite="rag", provider="bad", model="m1")
    with patch.object(
        runners,
        "_provider_for",
        side_effect=HTTPException(status_code=400, detail="bad provider"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await runners.run_eval_suite(payload)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "bad provider"


def test_validate_eval_suite_normalizes():
    assert runners.validate_eval_suite("RAG") == "rag"
    with pytest.raises(ValueError, match="Unsupported eval suite"):
        runners.validate_eval_suite("nope")


def test_summary_totals_from_fixture_attempts():
    attempts = [SimpleNamespace(total_tokens=5), SimpleNamespace(total_tokens=7)]
    fr = SimpleNamespace(
        estimated_cost_usd=0.1,
        total_tokens=None,
        attempts=attempts,
    )
    summary = SimpleNamespace(
        total_tokens=None, total_cost_usd=None, fixture_results=[fr]
    )
    tokens, cost = runners._summary_totals(summary)
    assert tokens == 12
    assert cost == pytest.approx(0.1)


def test_overall_score_prefers_known_attrs():
    assert runners._overall_score(SimpleNamespace(overall_score=0.9)) == 0.9
    assert runners._overall_score(SimpleNamespace()) is None


def test_provider_for_value_error_maps_to_http_400():
    with (
        patch.object(runners, "load_settings", return_value=_settings()),
        patch.object(
            runners,
            "_make_provider",
            side_effect=ValueError("unknown provider"),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            runners._provider_for("unknown", None)

    assert exc_info.value.status_code == 400
    assert "unknown provider" in str(exc_info.value.detail)


def test_summary_totals_uses_fixture_total_tokens():
    fr = SimpleNamespace(
        estimated_cost_usd=0.2,
        total_tokens=15,
        attempts=[],
    )
    summary = SimpleNamespace(
        total_tokens=None, total_cost_usd=None, fixture_results=[fr]
    )
    tokens, cost = runners._summary_totals(summary)
    assert tokens == 15
    assert cost == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_run_benchmark_invalid_tier_raises_http_400():
    payload = RunRequest(provider="ollama", tier="not-a-tier")
    with (
        patch.object(runners, "load_settings", return_value=_settings()),
        patch.object(runners, "_provider_for", return_value=MagicMock()),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await runners.run_benchmark_from_request(payload)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_run_benchmark_rethrows_http_exception():
    payload = RunRequest(provider="ollama", iterations=1)
    engine = MagicMock()
    engine.run = AsyncMock(
        side_effect=HTTPException(status_code=400, detail="engine boom")
    )
    repo = MagicMock()

    with (
        patch.object(runners, "load_settings", return_value=_settings()),
        patch.object(runners, "_provider_for", return_value=MagicMock()),
        patch("atomics.core.engine.LoopEngine", return_value=engine),
        patch("atomics.storage.repository.MetricsRepository", return_value=repo),
        patch(
            "atomics.tiers.get_tier_profile",
            return_value=SimpleNamespace(preferred_model=None),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await runners.run_benchmark_from_request(payload)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "engine boom"
    repo.close.assert_called_once()


@pytest.mark.asyncio
async def test_run_eval_from_request_rethrows_http_exception():
    payload = EvalRequest(suite="accuracy", provider="ollama")
    with patch.object(
        runners,
        "_provider_for",
        side_effect=HTTPException(status_code=400, detail="nope"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await runners.run_eval_from_request(payload)
    assert exc_info.value.detail == "nope"


@pytest.mark.asyncio
async def test_run_eval_from_request_value_error_maps_to_http_400():
    payload = EvalRequest(suite="accuracy", provider="ollama", model="m1")
    with (
        patch.object(runners, "_provider_for", side_effect=[MagicMock(), MagicMock()]),
        patch.object(
            runners, "run_eval", new_callable=AsyncMock, side_effect=ValueError("bad eval")
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await runners.run_eval_from_request(payload)

    assert exc_info.value.status_code == 400
    assert "bad eval" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_run_eval_suite_value_error_maps_to_http_400():
    payload = EvalRequest(suite="rag", provider="ollama", model="m1")
    with (
        patch.object(runners, "_provider_for", side_effect=[MagicMock(), MagicMock()]),
        patch.object(
            runners, "run_rag", new_callable=AsyncMock, side_effect=RuntimeError("rag fail")
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await runners.run_eval_suite(payload)

    assert exc_info.value.status_code == 400
    assert "rag fail" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_run_eval_suite_rethrows_http_exception_from_runner():
    payload = EvalRequest(suite="rag", provider="ollama", model="m1")
    with (
        patch.object(runners, "_provider_for", side_effect=[MagicMock(), MagicMock()]),
        patch.object(
            runners,
            "run_rag",
            new_callable=AsyncMock,
            side_effect=HTTPException(status_code=400, detail="inner"),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await runners.run_eval_suite(payload)

    assert exc_info.value.detail == "inner"

