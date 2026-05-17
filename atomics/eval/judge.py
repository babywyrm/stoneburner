"""LLM-as-judge: score a model response on accuracy, completeness, and format.

Design goals:
- Zero-cost by default: uses local Ollama so scoring never adds API spend.
- Rubric-based: structured 0-10 score that normalises to 0.0-1.0.
- Gold-criteria aware: fixtures can supply expected keywords/concepts so the
  judge knows what a correct answer looks like.
- Graceful degradation: parse failures return 0.5 (uncertain) rather than 0.

Rubric (max 10 points):
  Accuracy      0-4   Core content is factually correct and on-target
  Completeness  0-3   All key aspects of the question are addressed
  Format        0-3   Well-structured, clear, appropriate length
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from atomics.providers.base import BaseProvider

logger = logging.getLogger("atomics.eval.judge")

_JUDGE_SYSTEM = (
    "You are a strict technical reviewer. "
    "Score responses objectively — do not reward verbosity or penalise brevity "
    "if the content is correct and complete."
)

_RUBRIC_TEMPLATE = """\
TASK (what the model was asked):
{prompt}

{criteria_block}RESPONSE (what the model answered):
{response}

Score the response on three dimensions:
  Accuracy (0-4): Is the core content factually correct and on-target for the task?
  Completeness (0-3): Does it address all key aspects of the question?
  Format (0-3): Is it well-structured, readable, and an appropriate length?

Reply ONLY in this exact format — nothing else:
ACCURACY: <integer 0-4>
COMPLETENESS: <integer 0-3>
FORMAT: <integer 0-3>
RATIONALE: <one concise sentence explaining the score>
"""

_SCORE_RE = re.compile(
    r"ACCURACY:\s*(\d)\s*\nCOMPLETE\w*:\s*(\d)\s*\nFORMAT:\s*(\d)\s*\nRATIONALE:\s*(.+)",
    re.IGNORECASE,
)


@dataclass
class JudgeResult:
    score: float          # 0.0-1.0 normalised
    accuracy: int         # raw 0-4
    completeness: int     # raw 0-3
    format_score: int     # raw 0-3
    rationale: str
    judge_model: str
    parse_failed: bool = False


async def score_response(
    prompt: str,
    response: str,
    *,
    judge_provider: BaseProvider,
    judge_model: str | None = None,
    gold_criteria: list[str] | None = None,
    max_response_chars: int = 3000,
) -> JudgeResult:
    """Score a model response using an LLM judge.

    Args:
        prompt: The original task prompt sent to the model under test.
        response: The model's response to score.
        judge_provider: Provider to use for judging (default: local Ollama).
        judge_model: Model override for the judge.
        gold_criteria: Optional list of concepts/keywords a good answer should cover.
            Injected into the rubric as additional context for the judge.
        max_response_chars: Truncate responses beyond this to keep judge prompt lean.
    """
    truncated = response[:max_response_chars]
    if len(response) > max_response_chars:
        truncated += "\n[...response truncated for scoring...]"

    if gold_criteria:
        criteria_lines = "\n".join(f"  - {c}" for c in gold_criteria)
        criteria_block = (
            f"A good answer should cover these concepts:\n{criteria_lines}\n\n"
        )
    else:
        criteria_block = ""

    judge_prompt = _RUBRIC_TEMPLATE.format(
        prompt=prompt,
        criteria_block=criteria_block,
        response=truncated,
    )

    try:
        resp = await judge_provider.generate(
            judge_prompt,
            system=_JUDGE_SYSTEM,
            model=judge_model,
            max_tokens=128,
        )
        raw = resp.text.strip()
        effective_model = resp.model
    except Exception as exc:
        logger.warning("Judge call failed: %s", exc)
        return JudgeResult(
            score=0.5,
            accuracy=0,
            completeness=0,
            format_score=0,
            rationale=f"Judge call failed: {exc}",
            judge_model=judge_model or "unknown",
            parse_failed=True,
        )

    match = _SCORE_RE.search(raw)
    if not match:
        logger.warning("Judge parse failed for response: %r", raw[:200])
        return JudgeResult(
            score=0.5,
            accuracy=0,
            completeness=0,
            format_score=0,
            rationale=f"Parse failed: {raw[:100]}",
            judge_model=effective_model,
            parse_failed=True,
        )

    acc = min(int(match.group(1)), 4)
    comp = min(int(match.group(2)), 3)
    fmt = min(int(match.group(3)), 3)
    rationale = match.group(4).strip()
    raw_score = acc + comp + fmt          # 0-10
    normalised = round(raw_score / 10.0, 3)

    logger.debug(
        "Judge scored acc=%d comp=%d fmt=%d → %.3f | %s",
        acc, comp, fmt, normalised, rationale,
    )

    return JudgeResult(
        score=normalised,
        accuracy=acc,
        completeness=comp,
        format_score=fmt,
        rationale=rationale,
        judge_model=effective_model,
    )
