"""Quality-judge behaviour: truncation budget, scoring, parsing."""

from __future__ import annotations

import pytest

from atomics.eval.judge import (
    _parse_rubric,
    char_budget_for_tokens,
    compute_criteria_coverage,
    score_consensus,
    score_response,
)
from atomics.providers.base import ProviderResponse

_TRUNCATION_MARKER = "[...response truncated for scoring...]"
_VALID_RUBRIC = "ACCURACY: 4\nCOMPLETENESS: 3\nFORMAT: 3\nRATIONALE: solid answer."


class _PromptCapturingJudge:
    """Judge provider that records the prompt it was asked to score."""
    name = "fake-judge"

    def __init__(self, reply: str = _VALID_RUBRIC):
        self.reply = reply
        self.last_prompt = ""

    async def generate(self, prompt, *, system="", model=None, max_tokens=1024,
                        thinking=None, thinking_budget=None, temperature=None):
        self.last_prompt = prompt
        return ProviderResponse(
            text=self.reply, input_tokens=1, output_tokens=1, total_tokens=2,
            model="fake-judge", latency_ms=1.0, estimated_cost_usd=0.0,
        )

    async def health_check(self):
        return True


def test_char_budget_floor_for_short_fixtures():
    # Light fixtures keep the generous 3000-char floor.
    assert char_budget_for_tokens(256) == 3000
    assert char_budget_for_tokens(0) == 3000


def test_char_budget_scales_for_long_fixtures():
    # A 2000-token HEAVY fixture gets ~8000 chars instead of the old 3000 cap.
    assert char_budget_for_tokens(2000) == 8000
    assert char_budget_for_tokens(1500) == 6000


@pytest.mark.asyncio
async def test_long_response_truncated_with_default_budget():
    judge = _PromptCapturingJudge()
    long_response = "x" * 7000
    await score_response("q", long_response, judge_provider=judge)
    assert _TRUNCATION_MARKER in judge.last_prompt


@pytest.mark.asyncio
async def test_long_response_not_truncated_with_scaled_budget():
    judge = _PromptCapturingJudge()
    long_response = "x" * 7000
    await score_response(
        "q", long_response, judge_provider=judge,
        max_response_chars=char_budget_for_tokens(2000),  # 8000 >= 7000
    )
    assert _TRUNCATION_MARKER not in judge.last_prompt
    # Full response made it into the judge's view.
    assert long_response in judge.last_prompt


# ── Gold-criteria coverage anchor ───────────────────────────────────────────
def test_coverage_none_without_criteria():
    assert compute_criteria_coverage("anything", None) is None
    assert compute_criteria_coverage("anything", []) is None


def test_coverage_full_when_all_criteria_present():
    response = (
        "Shamir's secret sharing splits a secret using polynomial interpolation "
        "over a finite field; any threshold of shares reconstructs it."
    )
    criteria = ["polynomial interpolation", "finite field", "threshold of shares"]
    assert compute_criteria_coverage(response, criteria) == 1.0


def test_coverage_partial_when_some_criteria_missing():
    response = "It uses polynomial interpolation and nothing else worth noting."
    criteria = ["polynomial interpolation", "finite field", "threshold of shares"]
    # Only the first concept is present → 1/3.
    assert compute_criteria_coverage(response, criteria) == pytest.approx(0.333, abs=0.01)


def test_coverage_zero_for_empty_response():
    criteria = ["polynomial interpolation", "finite field"]
    assert compute_criteria_coverage("", criteria) == 0.0


def test_coverage_is_case_insensitive():
    response = "POLYNOMIAL INTERPOLATION over a FINITE FIELD."
    criteria = ["polynomial interpolation", "finite field"]
    assert compute_criteria_coverage(response, criteria) == 1.0


@pytest.mark.asyncio
async def test_score_response_attaches_coverage():
    judge = _PromptCapturingJudge()
    result = await score_response(
        "q", "polynomial interpolation over a finite field",
        judge_provider=judge,
        gold_criteria=["polynomial interpolation", "finite field"],
    )
    assert result.criteria_coverage == 1.0


@pytest.mark.asyncio
async def test_coverage_survives_judge_parse_failure():
    """Coverage is judge-independent: still computed when parsing fails."""
    judge = _PromptCapturingJudge(reply="garbage that does not match the rubric")
    result = await score_response(
        "q", "polynomial interpolation and finite field math",
        judge_provider=judge,
        gold_criteria=["polynomial interpolation", "finite field"],
    )
    assert result.parse_failed is True
    assert result.criteria_coverage == 1.0


# ── Multi-judge consensus + variance ────────────────────────────────────────
def _fixed_judge(name: str, reply: str):
    """A judge provider that always returns the same rubric reply."""
    class _J:
        def __init__(self):
            self.name = name

        async def generate(self, prompt, *, system="", model=None, max_tokens=1024,
                           thinking=None, thinking_budget=None, temperature=None):
            return ProviderResponse(
                text=reply, input_tokens=1, output_tokens=1, total_tokens=2,
                model=name, latency_ms=1.0, estimated_cost_usd=0.0,
            )

        async def health_check(self):
            return True
    return _J()


@pytest.mark.asyncio
async def test_consensus_single_judge_has_zero_stdev():
    judge = _PromptCapturingJudge()
    result = await score_consensus("q", "a", primary_judge=judge)
    assert result.n_judges == 1
    assert result.score_stdev == 0.0


@pytest.mark.asyncio
async def test_consensus_averages_scores_and_reports_stdev():
    # Judge A: 10/10 → 1.0 ; Judge B: 4/10 → 0.4. Mean 0.7, pstdev 0.3.
    judge_a = _fixed_judge("judge-a", "ACCURACY: 4\nCOMPLETENESS: 3\nFORMAT: 3\nRATIONALE: great.")
    judge_b = _fixed_judge("judge-b", "ACCURACY: 2\nCOMPLETENESS: 1\nFORMAT: 1\nRATIONALE: weak.")
    result = await score_consensus(
        "q", "a", primary_judge=judge_a, extra_judges=[(judge_b, None)],
    )
    assert result.n_judges == 2
    assert result.score == pytest.approx(0.7)
    assert result.score_stdev == pytest.approx(0.3)
    # Both judge names recorded.
    assert "judge-a" in result.judge_model and "judge-b" in result.judge_model


@pytest.mark.asyncio
async def test_consensus_excludes_failed_judges_from_mean():
    good = _fixed_judge("good", "ACCURACY: 4\nCOMPLETENESS: 3\nFORMAT: 3\nRATIONALE: ok.")
    broken = _fixed_judge("broken", "this is not a rubric")
    result = await score_consensus(
        "q", "a", primary_judge=good, extra_judges=[(broken, None)],
    )
    # Only the good judge counts.
    assert result.n_judges == 1
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_consensus_all_failed_returns_flagged_primary():
    broken_a = _fixed_judge("a", "nope")
    broken_b = _fixed_judge("b", "also nope")
    result = await score_consensus(
        "q", "a", primary_judge=broken_a, extra_judges=[(broken_b, None)],
    )
    assert result.parse_failed is True
    assert result.n_judges == 2


# ── Robust parsing + reformat retry + failure rate ──────────────────────────
def test_parse_rubric_strict_format():
    parsed = _parse_rubric("ACCURACY: 4\nCOMPLETENESS: 3\nFORMAT: 2\nRATIONALE: good.")
    assert parsed == (4, 3, 2, "good.")


def test_parse_rubric_tolerates_markdown_and_reordering():
    raw = (
        "Here is my assessment:\n"
        "**Format** - 2\n"
        "**Accuracy**: 4\n"
        "**Completeness** = 1\n"
        "Rationale: mostly correct but shallow."
    )
    parsed = _parse_rubric(raw)
    assert parsed is not None
    acc, comp, fmt, rationale = parsed
    assert (acc, comp, fmt) == (4, 1, 2)
    assert "shallow" in rationale


def test_parse_rubric_clamps_out_of_range():
    parsed = _parse_rubric("ACCURACY: 9\nCOMPLETENESS: 7\nFORMAT: 5\nRATIONALE: x")
    assert parsed == (4, 3, 3, "x")


def test_parse_rubric_returns_none_when_no_numbers():
    assert _parse_rubric("I cannot score this response, sorry.") is None


def test_parse_rubric_missing_rationale_is_tolerated():
    parsed = _parse_rubric("Accuracy 3 Completeness 2 Format 3")
    assert parsed is not None
    assert parsed[:3] == (3, 2, 3)


class _RetryJudge:
    """Returns a malformed reply first, then a clean one on the reformat retry."""
    name = "retry-judge"

    def __init__(self):
        self.calls = 0

    async def generate(self, prompt, *, system="", model=None, max_tokens=1024,
                        thinking=None, thinking_budget=None, temperature=None):
        self.calls += 1
        if self.calls == 1:
            text = "I think it's quite good overall, maybe four out of five stars."
        else:
            text = "ACCURACY: 4\nCOMPLETENESS: 3\nFORMAT: 3\nRATIONALE: reformatted."
        return ProviderResponse(
            text=text, input_tokens=1, output_tokens=1, total_tokens=2,
            model="retry-judge", latency_ms=1.0, estimated_cost_usd=0.0,
        )

    async def health_check(self):
        return True


@pytest.mark.asyncio
async def test_reformat_retry_recovers_unparseable_reply():
    judge = _RetryJudge()
    result = await score_response("q", "a", judge_provider=judge)
    assert judge.calls == 2  # one reformat retry happened
    assert result.parse_failed is False
    assert result.score == 1.0
    assert result.rationale == "reformatted."


@pytest.mark.asyncio
async def test_no_retry_when_first_reply_parses():
    judge = _RetryJudge()
    # Force a parseable first reply by pre-incrementing past the malformed branch.
    judge.calls = 1
    result = await score_response("q", "a", judge_provider=judge)
    # Only the single scoring call; calls goes 1 -> 2, no extra retry call.
    assert result.parse_failed is False
    assert judge.calls == 2


@pytest.mark.asyncio
async def test_parse_failure_rate_on_summary():
    from atomics.eval.judge import JudgeResult
    from atomics.eval.runner import EvalRunSummary, FixtureResult

    def _fr(parse_failed: bool):
        jr = JudgeResult(
            score=0.5 if parse_failed else 0.8, accuracy=0, completeness=0,
            format_score=0, rationale="", judge_model="m", parse_failed=parse_failed,
        )
        return FixtureResult(fixture=None, task_result=None, judge=jr)

    from datetime import UTC, datetime
    summary = EvalRunSummary(
        run_id="r", provider="p", model="m", judge_provider="j", judge_model="jm",
        started_at=datetime.now(UTC), completed_at=datetime.now(UTC),
        fixture_results=[_fr(True), _fr(False), _fr(False), _fr(False)],
    )
    assert summary.parse_failure_rate == 0.25
