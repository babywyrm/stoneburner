"""Quality-judge behaviour: truncation budget, scoring, parsing."""

from __future__ import annotations

import pytest

from atomics.eval.judge import (
    char_budget_for_tokens,
    compute_criteria_coverage,
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
