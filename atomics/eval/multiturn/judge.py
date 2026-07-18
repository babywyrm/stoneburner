"""Multi-turn conversation judge — scores per-turn quality and conversation coherence.

Per-turn rubric (max 10 points):
  Accuracy       0-4   Factually correct and on-target for this turn
  Context Use    0-3   Builds on and references prior conversation context
  Coherence      0-3   Consistent with previous answers, no contradictions

Conversation-level rubric (max 10 points):
  Retention      0-4   Remembers constraints/facts from earlier turns
  Consistency    0-3   No self-contradictions across the full conversation
  Instruction    0-3   Follows instructions that span multiple turns
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from atomics.providers.base import BaseProvider

logger = logging.getLogger("atomics.eval.multiturn.judge")

_TURN_JUDGE_SYSTEM = (
    "You are a strict conversation quality evaluator. "
    "Score how well each response uses prior conversation context."
)

_TURN_RUBRIC_TEMPLATE = """\
CONVERSATION SO FAR:
{transcript}

LATEST USER MESSAGE:
{user_message}

EXPECTED BEHAVIOR:
{expected_behavior}

MODEL RESPONSE TO SCORE:
{response}

Score the response on three dimensions:
  Accuracy (0-4): Is the content factually correct and on-target for this turn?
  Context_Use (0-3): Does it build on and reference prior conversation context appropriately?
  Coherence (0-3): Is it consistent with previous answers (no contradictions)?

Reply ONLY in this exact format:
ACCURACY: <integer 0-4>
CONTEXT_USE: <integer 0-3>
COHERENCE: <integer 0-3>
RATIONALE: <one concise sentence>
"""

_CONV_JUDGE_SYSTEM = (
    "You are a strict conversation quality evaluator. "
    "Score the overall conversation quality across all turns."
)

_CONV_RUBRIC_TEMPLATE = """\
FULL CONVERSATION:
{transcript}

CONVERSATION-LEVEL CRITERIA:
{criteria}

Score the model's overall performance across all turns:
  Retention (0-4): Does the model remember constraints and facts from earlier turns?
  Consistency (0-3): Are there any self-contradictions across the full conversation?
  Instruction (0-3): Does the model follow instructions that span multiple turns?

Reply ONLY in this exact format:
RETENTION: <integer 0-4>
CONSISTENCY: <integer 0-3>
INSTRUCTION: <integer 0-3>
RATIONALE: <one concise sentence>
"""

_TURN_RE = re.compile(
    r"ACCURACY:\s*(\d+)\s+CONTEXT.?USE:\s*(\d+)\s+COHERENCE:\s*(\d+)\s+RATIONALE:\s*([\s\S]+)",
    re.IGNORECASE,
)
_ACC_RE = re.compile(r"accuracy\W{0,6}(\d+)", re.IGNORECASE)
_CTX_RE = re.compile(r"context.?use\W{0,6}(\d+)", re.IGNORECASE)
_COH_RE = re.compile(r"coherence\W{0,6}(\d+)", re.IGNORECASE)

_CONV_RE = re.compile(
    r"RETENTION:\s*(\d+)\s+CONSISTENCY:\s*(\d+)\s+INSTRUCTION:\s*(\d+)\s+RATIONALE:\s*([\s\S]+)",
    re.IGNORECASE,
)
_RET_RE = re.compile(r"retention\W{0,6}(\d+)", re.IGNORECASE)
_CON_RE = re.compile(r"consistency\W{0,6}(\d+)", re.IGNORECASE)
_INS_RE = re.compile(r"instruction\W{0,6}(\d+)", re.IGNORECASE)
_RAT_RE = re.compile(r"rationale\W{0,6}([\s\S]+)", re.IGNORECASE)


@dataclass
class TurnJudgeResult:
    accuracy: int
    context_use: int
    coherence: int
    score: float
    rationale: str
    parse_failed: bool = False


@dataclass
class ConversationJudgeResult:
    retention: int
    consistency: int
    instruction: int
    score: float
    rationale: str
    parse_failed: bool = False


def _parse_turn_rubric(raw: str) -> tuple[int, int, int, str] | None:
    match = _TURN_RE.search(raw)
    if match:
        rationale = " ".join(match.group(4).strip().splitlines()).strip()
        return (min(int(match.group(1)), 4), min(int(match.group(2)), 3),
                min(int(match.group(3)), 3), rationale)

    a = _ACC_RE.search(raw)
    c = _CTX_RE.search(raw)
    h = _COH_RE.search(raw)
    if not (a and c and h):
        return None
    r = _RAT_RE.search(raw)
    rat = " ".join(r.group(1).strip().splitlines()).strip() if r else "(no rationale)"
    return (min(int(a.group(1)), 4), min(int(c.group(1)), 3),
            min(int(h.group(1)), 3), rat)


def _parse_conv_rubric(raw: str) -> tuple[int, int, int, str] | None:
    match = _CONV_RE.search(raw)
    if match:
        rationale = " ".join(match.group(4).strip().splitlines()).strip()
        return (min(int(match.group(1)), 4), min(int(match.group(2)), 3),
                min(int(match.group(3)), 3), rationale)

    r = _RET_RE.search(raw)
    c = _CON_RE.search(raw)
    i = _INS_RE.search(raw)
    if not (r and c and i):
        return None
    rat_m = _RAT_RE.search(raw)
    rat = " ".join(rat_m.group(1).strip().splitlines()).strip() if rat_m else "(no rationale)"
    return (min(int(r.group(1)), 4), min(int(c.group(1)), 3),
            min(int(i.group(1)), 3), rat)


async def score_turn(
    transcript: str,
    user_message: str,
    response: str,
    expected_behavior: str,
    judge: BaseProvider,
    *,
    judge_model: str | None = None,
) -> TurnJudgeResult:
    """Score a single conversation turn."""
    prompt = _TURN_RUBRIC_TEMPLATE.format(
        transcript=transcript,
        user_message=user_message,
        expected_behavior=expected_behavior,
        response=response[:3000],
    )
    try:
        resp = await judge.generate(
            prompt, system=_TURN_JUDGE_SYSTEM,
            model=judge_model, max_tokens=256, temperature=0.0,
        )
    except Exception:
        logger.warning("Turn judge call failed", exc_info=True)
        return TurnJudgeResult(0, 0, 0, 0.5, "judge call failed", parse_failed=True)

    parsed = _parse_turn_rubric(resp.text.strip())
    if parsed is None:
        return TurnJudgeResult(0, 0, 0, 0.5, "parse failed", parse_failed=True)

    acc, ctx, coh, rat = parsed
    return TurnJudgeResult(acc, ctx, coh, round((acc + ctx + coh) / 10.0, 3), rat)


async def score_conversation(
    transcript: str,
    criteria: list[str],
    judge: BaseProvider,
    *,
    judge_model: str | None = None,
) -> ConversationJudgeResult:
    """Score the overall conversation quality."""
    criteria_text = "\n".join(f"- {c}" for c in criteria) if criteria else "No specific criteria."
    prompt = _CONV_RUBRIC_TEMPLATE.format(
        transcript=transcript, criteria=criteria_text,
    )
    try:
        resp = await judge.generate(
            prompt, system=_CONV_JUDGE_SYSTEM,
            model=judge_model, max_tokens=256, temperature=0.0,
        )
    except Exception:
        logger.warning("Conversation judge call failed", exc_info=True)
        return ConversationJudgeResult(0, 0, 0, 0.5, "judge call failed", parse_failed=True)

    parsed = _parse_conv_rubric(resp.text.strip())
    if parsed is None:
        return ConversationJudgeResult(0, 0, 0, 0.5, "parse failed", parse_failed=True)

    ret, con, ins, rat = parsed
    return ConversationJudgeResult(ret, con, ins, round((ret + con + ins) / 10.0, 3), rat)
