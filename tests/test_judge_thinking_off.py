"""Judges ask for thinking off, so a reasoning judge answers in the format.

Left at the provider default, a model that reasons inline spends the small
judge token cap on prose and never reaches the score lines.
"""

from __future__ import annotations

import asyncio

from atomics.eval.multiturn.judge import score_conversation, score_turn
from atomics.eval.rag.fixtures import ALL_RAG_FIXTURES
from atomics.eval.rag.judge import score_rag_response
from tests.conftest import MockProvider


class _Recording(MockProvider):
    def __init__(self) -> None:
        super().__init__()
        self.thinking: list[bool | None] = []

    async def generate(self, prompt, *, thinking=None, **kwargs):  # type: ignore[no-untyped-def]
        self.thinking.append(thinking)
        return await super().generate(prompt, thinking=thinking, **kwargs)


def test_multiturn_and_rag_judges_turn_thinking_off():
    judge = _Recording()
    asyncio.run(score_turn("t", "u", "r", "e", judge))
    asyncio.run(score_conversation("t", ["c"], judge))
    asyncio.run(score_rag_response("r", ALL_RAG_FIXTURES[0], judge))
    assert judge.thinking == [False, False, False]
