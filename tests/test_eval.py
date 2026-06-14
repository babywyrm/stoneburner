"""Tests for the eval package: judge scoring, fixtures, and runner."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from atomics.eval.fixtures import EVAL_FIXTURES, EvalFixture
from atomics.eval.judge import JudgeResult, _SCORE_RE, score_response
from atomics.eval.runner import EvalRunSummary, run_eval
from atomics.models import TaskComplexity


# ── Fixtures ──────────────────────────────────────────────────────────────────


def test_fixture_ids_unique():
    ids = [f.id for f in EVAL_FIXTURES]
    assert len(ids) == len(set(ids)), "Duplicate fixture IDs"


def test_fixture_count():
    assert len(EVAL_FIXTURES) >= 15


def test_all_fixtures_have_gold_criteria():
    for f in EVAL_FIXTURES:
        assert f.gold_criteria, f"{f.id} is missing gold_criteria"


def test_fixture_complexity_spread():
    complexities = {f.complexity for f in EVAL_FIXTURES}
    assert TaskComplexity.LIGHT in complexities
    assert TaskComplexity.MODERATE in complexities
    assert TaskComplexity.HEAVY in complexities


def test_fixture_prompts_nonempty():
    for f in EVAL_FIXTURES:
        assert f.prompt.strip(), f"{f.id} has empty prompt"


# ── Judge regex ───────────────────────────────────────────────────────────────


def test_score_regex_parses_valid_output():
    raw = "ACCURACY: 3\nCOMPLETENESS: 2\nFORMAT: 3\nRATIONALE: Well-structured and accurate."
    m = _SCORE_RE.search(raw)
    assert m is not None
    assert int(m.group(1)) == 3
    assert int(m.group(2)) == 2
    assert int(m.group(3)) == 3
    assert "Well-structured" in m.group(4)


def test_score_regex_case_insensitive():
    raw = "accuracy: 4\ncompleteness: 3\nformat: 2\nrationale: Perfect."
    assert _SCORE_RE.search(raw) is not None


def test_score_regex_fails_on_missing_field():
    raw = "ACCURACY: 4\nCOMPLETENESS: 3\nRATIONALE: Missing format."
    assert _SCORE_RE.search(raw) is None


def test_score_regex_handles_crlf():
    """Judges behind Windows-style APIs may return CRLF — must not parse-fail."""
    raw = "ACCURACY: 3\r\nCOMPLETENESS: 2\r\nFORMAT: 2\r\nRATIONALE: CRLF response."
    m = _SCORE_RE.search(raw)
    assert m is not None
    assert int(m.group(1)) == 3


def test_score_regex_handles_completness_typo():
    """Qwen-14b occasionally misspells COMPLETENESS — the COMPLETE\\w* pattern absorbs it."""
    raw = "ACCURACY: 4\nCOMPLETNESS: 3\nFORMAT: 3\nRATIONALE: Typo in field name."
    m = _SCORE_RE.search(raw)
    assert m is not None
    assert int(m.group(2)) == 3


def test_score_regex_captures_multiline_rationale():
    """Multi-line rationales from verbose judges should be captured in full."""
    raw = (
        "ACCURACY: 2\nCOMPLETENESS: 1\nFORMAT: 2\n"
        "RATIONALE: The answer is partially correct.\nHowever, it omits key details."
    )
    m = _SCORE_RE.search(raw)
    assert m is not None
    # Both lines should appear in group(4)
    assert "partially correct" in m.group(4)
    assert "omits key details" in m.group(4)


# ── Judge scoring ─────────────────────────────────────────────────────────────


def _make_judge_provider(reply: str, model: str = "test-judge") -> MagicMock:
    from atomics.providers.base import ProviderResponse

    resp = ProviderResponse(
        text=reply,
        input_tokens=50,
        output_tokens=20,
        total_tokens=70,
        model=model,
        latency_ms=100.0,
        estimated_cost_usd=0.0,
    )
    provider = MagicMock()
    provider.generate = AsyncMock(return_value=resp)
    return provider


def test_score_response_perfect():
    reply = "ACCURACY: 4\nCOMPLETENESS: 3\nFORMAT: 3\nRATIONALE: Excellent response."
    provider = _make_judge_provider(reply)
    result = asyncio.run(
        score_response("What is SSRF?", "A detailed answer.", judge_provider=provider)
    )
    assert result.score == 1.0
    assert result.accuracy == 4
    assert result.completeness == 3
    assert result.format_score == 3
    assert not result.parse_failed


def test_score_response_low():
    reply = "ACCURACY: 1\nCOMPLETENESS: 0\nFORMAT: 1\nRATIONALE: Off-topic response."
    provider = _make_judge_provider(reply)
    result = asyncio.run(
        score_response("What is JWT?", "I don't know.", judge_provider=provider)
    )
    assert result.score == pytest.approx(0.2)
    assert not result.parse_failed


def test_score_response_parse_failure_returns_05():
    reply = "Sorry, I cannot score this."
    provider = _make_judge_provider(reply)
    result = asyncio.run(
        score_response("prompt", "response", judge_provider=provider)
    )
    assert result.score == 0.5
    assert result.parse_failed


def test_score_response_provider_exception_returns_05():
    provider = MagicMock()
    provider.generate = AsyncMock(side_effect=ConnectionError("Ollama down"))
    result = asyncio.run(
        score_response("prompt", "response", judge_provider=provider)
    )
    assert result.score == 0.5
    assert result.parse_failed


def test_score_response_truncates_long_response():
    provider = _make_judge_provider(
        "ACCURACY: 3\nCOMPLETENESS: 2\nFORMAT: 2\nRATIONALE: Truncated but scored."
    )
    long_response = "x" * 10_000
    result = asyncio.run(
        score_response("prompt", long_response, judge_provider=provider, max_response_chars=100)
    )
    # Check the generate call was made (truncation didn't break the call)
    provider.generate.assert_called_once()
    call_args = provider.generate.call_args
    assert "truncated" in call_args[0][0].lower() or len(call_args[0][0]) < 10_000


def test_score_response_injects_gold_criteria():
    provider = _make_judge_provider(
        "ACCURACY: 4\nCOMPLETENESS: 3\nFORMAT: 3\nRATIONALE: Covered all criteria."
    )
    gold = ["public/private key pair", "TLS handshake"]
    asyncio.run(
        score_response("Explain asymmetric encryption.", "Response.", judge_provider=provider, gold_criteria=gold)
    )
    call_prompt = provider.generate.call_args[0][0]
    assert "public/private key pair" in call_prompt
    assert "TLS handshake" in call_prompt


def test_score_normalisation():
    # 2 + 1 + 1 = 4 / 10 = 0.4
    reply = "ACCURACY: 2\nCOMPLETENESS: 1\nFORMAT: 1\nRATIONALE: Partial."
    provider = _make_judge_provider(reply)
    result = asyncio.run(score_response("p", "r", judge_provider=provider))
    assert result.score == pytest.approx(0.4)


# ── Eval runner ───────────────────────────────────────────────────────────────


def _make_test_provider(text: str = "A good answer.", cost: float = 0.001) -> MagicMock:
    from atomics.providers.base import ProviderResponse

    resp = ProviderResponse(
        text=text,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        model="test-model",
        latency_ms=200.0,
        estimated_cost_usd=cost,
    )
    provider = MagicMock()
    provider.name = "test"
    provider.generate = AsyncMock(return_value=resp)
    return provider


def _make_good_judge() -> MagicMock:
    return _make_judge_provider(
        "ACCURACY: 4\nCOMPLETENESS: 3\nFORMAT: 3\nRATIONALE: Good.", model="judge"
    )


def test_run_eval_returns_summary():
    provider = _make_test_provider()
    judge = _make_good_judge()
    summary = asyncio.run(run_eval(provider, judge_provider=judge))
    assert isinstance(summary, EvalRunSummary)
    assert len(summary.fixture_results) == len(EVAL_FIXTURES)


def test_run_eval_overall_accuracy():
    provider = _make_test_provider()
    judge = _make_good_judge()
    summary = asyncio.run(run_eval(provider, judge_provider=judge))
    assert summary.overall_accuracy is not None
    assert 0.0 <= summary.overall_accuracy <= 1.0


def test_run_eval_value_score_free_provider():
    """Free provider (cost=0) should have a finite value_score using the epsilon floor."""
    provider = _make_test_provider(cost=0.0)
    judge = _make_good_judge()
    summary = asyncio.run(run_eval(provider, judge_provider=judge))
    assert summary.value_score is not None
    assert summary.value_score > 0


def test_run_eval_on_fixture_done_called():
    provider = _make_test_provider()
    judge = _make_good_judge()
    calls = []
    summary = asyncio.run(
        run_eval(provider, judge_provider=judge, on_fixture_done=lambda fr: calls.append(fr))
    )
    assert len(calls) == len(summary.fixture_results)


def test_run_eval_provider_failure_recorded():
    provider = MagicMock()
    provider.name = "failing"
    provider.generate = AsyncMock(side_effect=ConnectionError("down"))
    judge = _make_good_judge()
    summary = asyncio.run(run_eval(provider, judge_provider=judge))
    failed = [r for r in summary.fixture_results if r.task_result.status.value == "failed"]
    assert len(failed) == len(EVAL_FIXTURES)
    assert summary.overall_accuracy is None


def test_run_eval_empty_exception_message_falls_back_to_repr():
    """A failure whose str(exc) is empty (e.g. httpx.ReadTimeout) must still
    record a non-blank error_message via repr, not an empty string."""
    import httpx

    provider = MagicMock()
    provider.name = "timeouty"
    provider.generate = AsyncMock(side_effect=httpx.ReadTimeout(""))
    judge = _make_good_judge()
    summary = asyncio.run(
        run_eval(provider, judge_provider=judge, fixtures=[EVAL_FIXTURES[0]])
    )
    tr = summary.fixture_results[0].task_result
    assert tr.status.value == "failed"
    assert tr.error_class == "ReadTimeout"
    assert tr.error_message  # non-empty
    assert "ReadTimeout" in tr.error_message


def test_run_eval_on_fixture_done_called_for_failures():
    """Callback must fire even when the provider under test raises an exception.

    Regression: before the fix the ``continue`` after the failure block skipped
    the callback, so failed fixtures were invisible in the live CLI table and
    were never saved to the database.
    """
    provider = MagicMock()
    provider.name = "failing"
    provider.generate = AsyncMock(side_effect=ConnectionError("down"))
    judge = _make_good_judge()
    calls = []
    summary = asyncio.run(
        run_eval(provider, judge_provider=judge, on_fixture_done=lambda fr: calls.append(fr))
    )
    # Every fixture (all failed) must have triggered the callback
    assert len(calls) == len(EVAL_FIXTURES)
    assert all(c.task_result.status.value == "failed" for c in calls)


# ── TaskResult model ─────────────────────────────────────────────────────────


def test_task_result_has_accuracy_fields():
    from atomics.models import TaskCategory, TaskResult

    r = TaskResult(
        run_id="test",
        category=TaskCategory.GENERAL_QA,
        task_name="ev-01",
        provider="ollama",
        model="qwen2.5:7b",
    )
    assert r.accuracy_score is None
    assert r.judge_model == ""
    assert r.quality_rationale == ""


def test_task_result_accuracy_fields_settable():
    from atomics.models import TaskCategory, TaskResult

    r = TaskResult(
        run_id="test",
        category=TaskCategory.GENERAL_QA,
        task_name="ev-01",
        provider="claude",
        model="claude-sonnet-4",
        accuracy_score=0.9,
        judge_model="qwen2.5:7b",
        quality_rationale="Well covered.",
    )
    assert r.accuracy_score == pytest.approx(0.9)
    assert r.judge_model == "qwen2.5:7b"
