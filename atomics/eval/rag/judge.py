"""RAG-specific LLM-as-judge: grounding, faithfulness, and abstention scoring.

Rubric (max 10 points):
  Grounding     0-4   Answer references and uses the provided context
  Faithfulness  0-3   Stays within what the context says (no hallucination)
  Abstention    0-3   Correctly declines when context lacks the answer

Normalized score = (grounding + faithfulness + abstention) / 10.0
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from atomics.providers.base import BaseProvider

if TYPE_CHECKING:
    from atomics.eval.rag import RAGFixture

logger = logging.getLogger("atomics.eval.rag.judge")

_RAG_JUDGE_SYSTEM = (
    "You are a strict RAG quality evaluator. "
    "Score responses based on how well they use the provided context. "
    "Do not reward answers that go beyond the context, even if factually correct."
)

_RAG_RUBRIC_TEMPLATE = """\
QUESTION:
{question}

CONTEXT PROVIDED TO THE MODEL:
{context}

EXPECTED BEHAVIOR:
{behavior}

MODEL RESPONSE:
{response}

Score the response on three dimensions:
  Grounding (0-4): Does the answer reference and use the provided context? \
A score of 0 means the answer ignores the context entirely. \
A score of 4 means every claim is clearly grounded in specific parts of the context.
  Faithfulness (0-3): Does the answer stay within what the context actually says? \
Penalise any claims, facts, or details not present in the context (hallucinations). \
A score of 0 means major fabrication. A score of 3 means fully faithful.
  Abstention (0-3): {abstention_guidance}

Reply ONLY in this exact format — nothing else:
GROUNDING: <integer 0-4>
FAITHFULNESS: <integer 0-3>
ABSTENTION: <integer 0-3>
RATIONALE: <one concise sentence explaining the scores>
"""

_ABSTENTION_HAS_ANSWER = (
    "The context DOES contain the answer. A score of 3 means the model "
    "correctly provided an answer. A score of 0 means it incorrectly refused "
    "to answer or claimed the information was not available."
)

_ABSTENTION_NO_ANSWER = (
    "The context does NOT contain the answer. A score of 3 means the model "
    "correctly stated it cannot answer from the given context. A score of 0 "
    "means it fabricated an answer instead of declining."
)

_SCORE_RE = re.compile(
    r"GROUNDING:\s*(\d+)\s+"
    r"FAITHFUL\w*:\s*(\d+)\s+"
    r"ABSTENTION:\s*(\d+)\s+"
    r"RATIONALE:\s*([\s\S]+)",
    re.IGNORECASE,
)

_GROUNDING_RE = re.compile(r"grounding\W{0,6}(\d+)", re.IGNORECASE)
_FAITHFUL_RE = re.compile(r"faithful\w*\W{0,6}(\d+)", re.IGNORECASE)
_ABSTENTION_RE = re.compile(r"abstention\W{0,6}(\d+)", re.IGNORECASE)
_RATIONALE_RE = re.compile(r"rationale\W{0,6}([\s\S]+)", re.IGNORECASE)


@dataclass
class RAGJudgeResult:
    """Parsed result from the RAG judge."""

    grounding: int
    faithfulness: int
    abstention: int
    score: float  # normalized 0.0–1.0
    rationale: str
    parse_failed: bool = False
    raw: str = ""


def _parse_rag_rubric(raw: str) -> tuple[int, int, int, str] | None:
    """Parse a RAG judge reply into (grounding, faithfulness, abstention, rationale)."""
    match = _SCORE_RE.search(raw)
    if match:
        rationale = " ".join(match.group(4).strip().splitlines()).strip()
        return (
            min(int(match.group(1)), 4),
            min(int(match.group(2)), 3),
            min(int(match.group(3)), 3),
            rationale,
        )

    g_m = _GROUNDING_RE.search(raw)
    f_m = _FAITHFUL_RE.search(raw)
    a_m = _ABSTENTION_RE.search(raw)
    if not (g_m and f_m and a_m):
        return None

    rat_m = _RATIONALE_RE.search(raw)
    rationale = (
        " ".join(rat_m.group(1).strip().splitlines()).strip()
        if rat_m
        else "(no rationale provided)"
    )
    return (
        min(int(g_m.group(1)), 4),
        min(int(f_m.group(1)), 3),
        min(int(a_m.group(1)), 3),
        rationale,
    )


def _format_context(chunks: list) -> str:
    """Render context chunks for the judge prompt."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Document {i}: {chunk.source}]\n{chunk.content}")
    return "\n\n".join(parts)


async def score_rag_response(
    response: str,
    fixture: RAGFixture,
    judge: BaseProvider,
    *,
    judge_model: str | None = None,
    max_response_chars: int = 4000,
) -> RAGJudgeResult:
    """Score a RAG response using the grounding/faithfulness/abstention rubric."""
    truncated = response[:max_response_chars]
    context_text = _format_context(fixture.context_chunks)

    if fixture.context_contains_answer:
        behavior = (
            "The context contains the answer. The model should provide a "
            "grounded, faithful response using the context."
        )
        abstention_guidance = _ABSTENTION_HAS_ANSWER
    else:
        behavior = (
            "The context does NOT contain the answer. The model should "
            "decline to answer or state the information is not available."
        )
        abstention_guidance = _ABSTENTION_NO_ANSWER

    prompt = _RAG_RUBRIC_TEMPLATE.format(
        question=fixture.question,
        context=context_text,
        behavior=behavior,
        response=truncated,
        abstention_guidance=abstention_guidance,
    )

    try:
        judge_resp = await judge.generate(
            prompt,
            system=_RAG_JUDGE_SYSTEM,
            model=judge_model,
            max_tokens=256,
            temperature=0.0,
        )
    except Exception:
        logger.warning("RAG judge call failed", exc_info=True)
        return RAGJudgeResult(
            grounding=0,
            faithfulness=0,
            abstention=0,
            score=0.5,
            rationale="judge call failed",
            parse_failed=True,
        )

    raw = judge_resp.text.strip()
    parsed = _parse_rag_rubric(raw)

    if parsed is None:
        logger.warning("RAG judge parse failed: %s", raw[:200])
        return RAGJudgeResult(
            grounding=0,
            faithfulness=0,
            abstention=0,
            score=0.5,
            rationale="parse failed",
            parse_failed=True,
            raw=raw,
        )

    grounding, faithfulness, abstention, rationale = parsed
    score = (grounding + faithfulness + abstention) / 10.0

    return RAGJudgeResult(
        grounding=grounding,
        faithfulness=faithfulness,
        abstention=abstention,
        score=round(score, 3),
        rationale=rationale,
        raw=raw,
    )


def compute_hallucination(
    response: str,
    fixture: RAGFixture,
) -> bool:
    """Heuristic: did the response contain claims not grounded in any chunk?

    Returns True if the response appears to contain fabricated content.
    This is a conservative lexical check — the judge score is the primary signal.
    """
    if not fixture.context_contains_answer:
        lower = response.lower()
        decline_phrases = [
            "not available",
            "not in the context",
            "cannot answer",
            "don't have",
            "no information",
            "not provided",
            "not mentioned",
            "unable to find",
            "outside the provided",
        ]
        if any(p in lower for p in decline_phrases):
            return False
        return len(response.strip()) > 50

    return False
