"""Scoring for archreview: objective category match, robustness, judge reasoning."""

from __future__ import annotations

import re
import statistics

from atomics.archreview.models import AnswerKey, Finding
from atomics.providers.base import BaseProvider


def _harmonic(a: float, b: float) -> float:
    return round(2 * a * b / (a + b), 4) if (a + b) > 0 else 0.0


def score_objective(
    findings: list[Finding], answer_key: AnswerKey,
) -> tuple[float, float, float, list[str]]:
    """Return (weighted_recall, precision, f, matched_categories).

    - weighted_recall: Σ weight(matched present cats) / Σ weight(all present)
    - precision: |emitted ∩ present| / |emitted real categories|
      ("unknown" findings are excluded from the precision denominator)
    """
    present = answer_key.weights
    emitted = {f.category for f in findings if f.category != "unknown"}
    matched = sorted(c for c in emitted if c in present)

    total_w = answer_key.total_weight()
    matched_w = sum(present[c] for c in matched)
    recall = round(matched_w / total_w, 4) if total_w > 0 else 0.0
    precision = round(len(matched) / len(emitted), 4) if emitted else 0.0
    fscore = _harmonic(recall, precision)
    return recall, precision, fscore, matched


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def compute_robustness(
    round_category_sets: list[set[str]], round_recalls: list[float],
) -> tuple[float, float]:
    """Return (mean_pairwise_jaccard_stability, recall_stdev) across rounds.

    Stability is the mean Jaccard similarity over all round pairs (1.0 = the
    model emitted the same category set every round). Single round → (1.0, 0.0).
    """
    n = len(round_category_sets)
    if n <= 1:
        return 1.0, 0.0
    sims = [
        _jaccard(round_category_sets[i], round_category_sets[j])
        for i in range(n) for j in range(i + 1, n)
    ]
    stability = round(sum(sims) / len(sims), 4)
    recall_sd = round(statistics.pstdev(round_recalls), 4) if len(round_recalls) > 1 else 0.0
    return stability, recall_sd


_REASONING_SYSTEM = (
    "You are a strict reviewer of security-architecture analyses. "
    "Judge the reasoning quality, not formatting."
)

_REASONING_TEMPLATE = """\
A model produced the following security-architecture analysis of a codebase.
Rate the QUALITY OF ITS REASONING from 0-10 on: trust-boundary identification,
data-flow correctness, prioritization sanity, and explanation accuracy.

Reply ONLY as:
REASONING: <integer 0-10>
RATIONALE: <one sentence>

ANALYSIS:
{analysis}
"""

_REASONING_RE = re.compile(
    r"REASONING\W{0,4}(\d+)[\s\S]*?RATIONALE\W{0,4}([\s\S]+)", re.IGNORECASE)


async def score_reasoning(
    analysis_text: str, *, judge: BaseProvider, judge_model: str | None,
) -> tuple[float, str]:
    """Judge architectural reasoning 0-10 → normalized 0.0-1.0. (0.5 on parse fail)."""
    prompt = _REASONING_TEMPLATE.format(analysis=analysis_text[:8000])
    resp = await judge.generate(prompt, system=_REASONING_SYSTEM,
                                model=judge_model, max_tokens=256,
                                thinking=False, temperature=0.0)
    m = _REASONING_RE.search(resp.text)
    if not m:
        return 0.5, "(unparseable judge reply)"
    raw = min(int(m.group(1)), 10)
    rationale = " ".join(m.group(2).strip().splitlines()).strip()
    return round(raw / 10.0, 4), rationale
