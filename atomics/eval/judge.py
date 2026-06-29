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
import statistics
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
    # [\r\n]+ handles both CRLF (Windows / some APIs) and LF.
    # COMPLET\w* absorbs both "COMPLETENESS" and "COMPLETNESS" (qwen typo).
    # RATIONALE uses [\s\S]+ so multi-line rationales are captured in full.
    r"ACCURACY:\s*(\d+)[\r\n]+COMPLET\w*:\s*(\d+)[\r\n]+FORMAT:\s*(\d+)[\r\n]+RATIONALE:\s*([\s\S]+)",
    re.IGNORECASE,
)

# Lenient field-by-field patterns for the fallback parser. Each tolerates
# markdown bold/markers, arbitrary separators, and field reordering — e.g.
# "**Accuracy** - 4", "ACCURACY = 4", "Accuracy: 4/4".
_ACC_RE = re.compile(r"accuracy\W{0,6}(\d+)", re.IGNORECASE)
_COMP_RE = re.compile(r"complet\w*\W{0,6}(\d+)", re.IGNORECASE)
_FMT_RE = re.compile(r"format\W{0,6}(\d+)", re.IGNORECASE)
_RATIONALE_RE = re.compile(r"rationale\W{0,6}([\s\S]+)", re.IGNORECASE)

_REFORMAT_SYSTEM = (
    "You reformat a previous evaluation into the exact required format. "
    "Do not change the scores; only restructure them."
)

_REFORMAT_TEMPLATE = """\
Your previous reply could not be parsed. Re-emit the SAME scores using ONLY
these four lines and nothing else (no preamble, no markdown):
ACCURACY: <integer 0-4>
COMPLETENESS: <integer 0-3>
FORMAT: <integer 0-3>
RATIONALE: <one concise sentence>

Your previous reply was:
{bad}
"""


def _parse_rubric(raw: str) -> tuple[int, int, int, str] | None:
    """Parse a judge reply into (accuracy, completeness, format, rationale).

    Tries the strict line-oriented format first, then falls back to a lenient
    field-by-field scan that tolerates markdown, reordering, and stray prose.
    Returns None only when the three numeric scores cannot be recovered.
    """
    match = _SCORE_RE.search(raw)
    if match:
        rationale = " ".join(match.group(4).strip().splitlines()).strip()
        return (
            min(int(match.group(1)), 4),
            min(int(match.group(2)), 3),
            min(int(match.group(3)), 3),
            rationale,
        )

    acc_m = _ACC_RE.search(raw)
    comp_m = _COMP_RE.search(raw)
    fmt_m = _FMT_RE.search(raw)
    if not (acc_m and comp_m and fmt_m):
        return None

    rat_m = _RATIONALE_RE.search(raw)
    rationale = (
        " ".join(rat_m.group(1).strip().splitlines()).strip()
        if rat_m else "(no rationale provided)"
    )
    return (
        min(int(acc_m.group(1)), 4),
        min(int(comp_m.group(1)), 3),
        min(int(fmt_m.group(1)), 3),
        rationale,
    )


# Rough chars-per-token for sizing the judge's view of a response. English text
# averages ~4 chars/token; we use this to scale truncation to a fixture's
# expected output length so long answers aren't cut off (which would unfairly
# penalise Completeness on MODERATE/HEAVY fixtures).
_CHARS_PER_TOKEN = 4
_MIN_JUDGE_CHARS = 3000


def char_budget_for_tokens(max_output_tokens: int) -> int:
    """Judge-visible character budget for a response of up to N output tokens.

    Floored at _MIN_JUDGE_CHARS so short fixtures keep generous headroom, and
    scaled by chars/token so a 2000-token answer is judged in full rather than
    truncated at the old fixed 3000-char cap.
    """
    return max(_MIN_JUDGE_CHARS, max_output_tokens * _CHARS_PER_TOKEN)


# Words ignored when matching a gold criterion's key terms against a response.
_COVERAGE_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "to", "in", "is", "for", "with", "on",
    "how", "does", "what", "why", "be", "are", "as", "by", "it", "that", "this",
})
# Fraction of a criterion's significant terms that must appear for it to count
# as covered. Lets multi-word concepts tolerate a missing minor word.
_COVERAGE_TERM_THRESHOLD = 0.6


def _criterion_covered(criterion: str, response_lower: str) -> bool:
    terms = [
        t for t in re.findall(r"[a-z0-9]+", criterion.lower())
        if t not in _COVERAGE_STOPWORDS and len(t) > 2
    ]
    if not terms:
        return criterion.lower().strip() in response_lower
    hits = sum(1 for t in terms if t in response_lower)
    return hits / len(terms) >= _COVERAGE_TERM_THRESHOLD


def detect_self_judge(
    under_test: BaseProvider,
    under_test_model: str | None,
    judges: list[tuple[BaseProvider, str | None]],
) -> list[str]:
    """Return labels of judges that are the same provider+model as the test model.

    Same-model judging suffers self-preference bias (a model rates its own style
    of output higher), so callers should warn when this is detected. A judge
    collides when it shares the provider name AND resolves to the same model as
    the model under test (an unspecified model resolves to the provider default).
    """
    ut_model = under_test_model or under_test.default_model
    if ut_model is None:
        return []
    collisions: list[str] = []
    for judge_provider, judge_model in judges:
        j_model = judge_model or judge_provider.default_model
        if judge_provider.name == under_test.name and j_model == ut_model:
            collisions.append(f"{judge_provider.name}:{j_model}")
    return collisions


def compute_criteria_coverage(
    response: str, gold_criteria: list[str] | None,
) -> float | None:
    """Objective, judge-independent fraction of gold criteria present in a response.

    A lexical anchor (not semantic): each criterion counts as covered when a
    majority of its significant terms appear in the response. Returns None when
    a fixture supplies no gold criteria. This complements the LLM judge's
    advisory use of the same criteria with a deterministic signal that can't be
    gamed by a verbose-but-empty answer.
    """
    if not gold_criteria:
        return None
    response_lower = response.lower()
    covered = sum(1 for c in gold_criteria if _criterion_covered(c, response_lower))
    return round(covered / len(gold_criteria), 3)


@dataclass
class JudgeResult:
    score: float          # 0.0-1.0 normalised
    accuracy: int         # raw 0-4
    completeness: int     # raw 0-3
    format_score: int     # raw 0-3
    rationale: str
    judge_model: str
    parse_failed: bool = False
    criteria_coverage: float | None = None  # lexical gold-criteria coverage, 0.0-1.0
    score_stdev: float | None = None  # inter-judge stdev when scored by a panel
    n_judges: int = 1  # number of judges whose scores were aggregated


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
    # Deterministic anchor over the *full* response (independent of judge and of
    # the prompt truncation below).
    criteria_coverage = compute_criteria_coverage(response, gold_criteria)

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

    async def _ask(text: str, system: str) -> tuple[str, str]:
        """Ask the judge model and return (text, model_name).

        Strategy for thinking-capable models:
        1. First attempt: thinking=False for a direct structured response.
        2. If the response is empty (model requires thinking to produce output),
           retry with thinking enabled and use the visible response.
        3. If still empty but thinking_text has content, attempt to parse scores
           from the thinking output as a last resort.

        This handles qwen3.6, deepseek-r1, gemma4, and future thinking models
        without hardcoding model-specific behavior.
        """
        resp = await judge_provider.generate(
            text,
            system=system,
            model=judge_model,
            max_tokens=128,
            temperature=0.0,
            thinking=False,
        )
        result_text = resp.text.strip()

        if result_text:
            return result_text, resp.model

        # Fallback: response was empty — try with thinking enabled (some models
        # need to "think" before they can produce any output at all).
        logger.info("Judge returned empty with thinking=False; retrying with thinking enabled.")
        resp = await judge_provider.generate(
            text,
            system=system,
            model=judge_model,
            max_tokens=512,
            temperature=0.0,
            thinking=True,
            thinking_budget=256,
        )
        result_text = resp.text.strip()

        if result_text:
            return result_text, resp.model

        # Last resort: check if scores are embedded in the thinking output.
        thinking_text = getattr(resp, "thinking_text", "") or ""
        if thinking_text:
            logger.info("Judge visible response empty; attempting parse from thinking content.")
            return thinking_text.strip(), resp.model

        return "", resp.model

    try:
        raw, effective_model = await _ask(judge_prompt, _JUDGE_SYSTEM)
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
            criteria_coverage=criteria_coverage,
        )

    parsed = _parse_rubric(raw)

    # One reformat retry: the scores are often present but mis-formatted (markdown,
    # preamble, reordering). Ask the judge to re-emit the same scores cleanly
    # before giving up.
    if parsed is None:
        logger.info("Judge reply unparseable; attempting one reformat retry.")
        try:
            retry_raw, effective_model = await _ask(
                _REFORMAT_TEMPLATE.format(bad=raw[:500]), _REFORMAT_SYSTEM,
            )
            parsed = _parse_rubric(retry_raw)
            if parsed is not None:
                raw = retry_raw
        except Exception as exc:
            logger.warning("Judge reformat retry failed: %s", exc)

    if parsed is None:
        logger.warning("Judge parse failed for response: %r", raw[:200])
        return JudgeResult(
            score=0.5,
            accuracy=0,
            completeness=0,
            format_score=0,
            rationale=f"Parse failed: {raw[:100]}",
            judge_model=effective_model,
            parse_failed=True,
            criteria_coverage=criteria_coverage,
        )

    acc, comp, fmt, rationale = parsed
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
        criteria_coverage=criteria_coverage,
    )


async def score_consensus(
    prompt: str,
    response: str,
    *,
    primary_judge: BaseProvider,
    primary_model: str | None = None,
    extra_judges: list[tuple[BaseProvider, str | None]] | None = None,
    gold_criteria: list[str] | None = None,
    max_response_chars: int = 3000,
) -> JudgeResult:
    """Score a response with a panel of judges and aggregate to a consensus.

    Runs the primary judge plus any extra (provider, model) judges, averages the
    normalised scores of the judges that parsed successfully, and reports the
    inter-judge standard deviation as score_stdev — a measure of how much the
    judges disagree. Falls back to the primary judge's result when there is no
    panel or when every judge fails to parse.
    """
    extra_judges = extra_judges or []

    results: list[JudgeResult] = [
        await score_response(
            prompt, response,
            judge_provider=primary_judge, judge_model=primary_model,
            gold_criteria=gold_criteria, max_response_chars=max_response_chars,
        )
    ]
    for jp, jm in extra_judges:
        results.append(
            await score_response(
                prompt, response,
                judge_provider=jp, judge_model=jm,
                gold_criteria=gold_criteria, max_response_chars=max_response_chars,
            )
        )

    # criteria_coverage is judge-independent, so it's identical across results.
    coverage = results[0].criteria_coverage

    valid = [r for r in results if not r.parse_failed]
    if not valid:
        # Everyone failed to parse; surface the primary result, flagged.
        primary = results[0]
        primary.n_judges = len(results)
        primary.score_stdev = 0.0
        return primary

    scores = [r.score for r in valid]
    mean_score = round(sum(scores) / len(scores), 3)
    stdev = round(statistics.pstdev(scores), 3) if len(scores) > 1 else 0.0

    return JudgeResult(
        score=mean_score,
        accuracy=round(statistics.mean(r.accuracy for r in valid)),
        completeness=round(statistics.mean(r.completeness for r in valid)),
        format_score=round(statistics.mean(r.format_score for r in valid)),
        # Primary judge's rationale is the representative explanation.
        rationale=valid[0].rationale,
        judge_model=", ".join(r.judge_model for r in valid),
        criteria_coverage=coverage,
        score_stdev=stdev,
        n_judges=len(valid),
    )
