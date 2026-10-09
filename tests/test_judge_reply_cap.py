"""Every judge's first call leaves room for a reasoning model to think.

gpt-oss ignores ``thinking=False``. At 256 tokens its conversation judge ran
out while still thinking, so the multiturn score came back blank.
"""

from __future__ import annotations

import pytest

from atomics.archreview.scorer import score_reasoning
from atomics.eval.adversarial.scorer import score_resistance
from atomics.eval.judge import JUDGE_MAX_TOKENS, score_response
from atomics.eval.multiturn.judge import score_conversation, score_turn
from atomics.eval.rag.fixtures import RAG_FIXTURES
from atomics.eval.rag.judge import score_rag_response
from atomics.providers.base import BaseProvider, ProviderResponse


class _Recorder(BaseProvider):
    name = "recorder"

    def __init__(self) -> None:
        self.caps: list[int] = []

    async def generate(self, prompt, *, max_tokens=1024, **kwargs) -> ProviderResponse:
        self.caps.append(max_tokens)
        return ProviderResponse("unparsable", 1, 1, 2, "judge", 1.0, 0.0)

    async def health_check(self) -> bool:
        return True


JUDGES = {
    "eval": lambda j: score_response("p", "r", judge_provider=j),
    "rag": lambda j: score_rag_response("r", RAG_FIXTURES[0], j),
    "turn": lambda j: score_turn("", "u", "r", "e", j),
    "conversation": lambda j: score_conversation("t", [], j),
    "archreview": lambda j: score_reasoning("a", judge=j, judge_model=None),
    "adversarial": lambda j: score_resistance(
        "p", "r", attack_goal="g", resistance_criteria=[], judge_provider=j
    ),
}


@pytest.mark.parametrize("judge", JUDGES)
async def test_first_judge_call_uses_the_shared_cap(judge: str) -> None:
    recorder = _Recorder()
    await JUDGES[judge](recorder)
    assert recorder.caps[0] == JUDGE_MAX_TOKENS >= 1024
