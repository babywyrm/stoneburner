"""Quality-judge behaviour: truncation budget, scoring, parsing."""

from __future__ import annotations

import pytest

from atomics.eval.judge import (
    char_budget_for_tokens,
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
