"""Judge calibration: verify the quality judge ranks graded answers correctly.

A judge is only trustworthy if a clearly-better answer reliably scores higher
than a clearly-worse one. This module ships a small set of fixtures, each with
several answers ordered worst → best, and a harness that scores them and checks
that the resulting scores are monotonically non-decreasing and well separated.

Used as a regression guard (see tests/test_calibration.py): a judge-prompt or
parsing change that breaks ranking will fail the calibration check.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from atomics.eval.judge import JudgeResult, char_budget_for_tokens, score_response
from atomics.providers.base import BaseProvider

# Signature of a scoring function compatible with score_response.
ScoreFn = Callable[..., Awaitable[JudgeResult]]


@dataclass(frozen=True)
class CalibrationCase:
    """A prompt with answers ordered from worst (index 0) to best (index -1)."""
    id: str
    prompt: str
    graded_answers: tuple[str, ...]
    gold_criteria: tuple[str, ...] = ()


# Each case's answers are deliberately separated by quality: an off-topic/wrong
# answer, a thin-but-correct answer, and a thorough, technically precise answer.
CALIBRATION_CASES: tuple[CalibrationCase, ...] = (
    CalibrationCase(
        id="cal-hash",
        prompt="What is a cryptographic hash function and name one security property?",
        graded_answers=(
            "A hash function is a kind of breakfast you make with potatoes.",
            "A hash function maps data to a fixed-size value.",
            (
                "A cryptographic hash function deterministically maps arbitrary-length "
                "input to a fixed-size digest. Key security properties include "
                "preimage resistance, second-preimage resistance, and collision "
                "resistance, so it is infeasible to invert or find colliding inputs."
            ),
        ),
        gold_criteria=("fixed-size digest", "collision resistance", "preimage resistance"),
    ),
    CalibrationCase(
        id="cal-tcp",
        prompt="Explain what TCP provides to applications.",
        graded_answers=(
            "TCP is a popular brand of coffee enjoyed worldwide.",
            "TCP is a networking protocol that sends data between computers.",
            (
                "TCP is a connection-oriented transport protocol that provides "
                "reliable, ordered, error-checked delivery of a byte stream. It uses "
                "a three-way handshake, sequence numbers and acknowledgements, "
                "retransmission of lost segments, and flow and congestion control."
            ),
        ),
        gold_criteria=("connection-oriented", "reliable", "congestion control"),
    ),
)


@dataclass
class CaseCalibration:
    case_id: str
    scores: list[float]
    monotonic: bool       # scores never decrease from worst → best
    separated: bool       # best - worst >= min_separation


@dataclass
class CalibrationReport:
    results: list[CaseCalibration] = field(default_factory=list)
    min_separation: float = 0.2

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(
            r.monotonic and r.separated for r in self.results
        )

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        ok = sum(1 for r in self.results if r.monotonic and r.separated)
        return round(ok / len(self.results), 3)


def _is_monotonic(scores: Sequence[float]) -> bool:
    return all(b >= a for a, b in zip(scores, scores[1:]))


async def calibrate_judge(
    judge_provider: BaseProvider,
    *,
    judge_model: str | None = None,
    cases: Sequence[CalibrationCase] = CALIBRATION_CASES,
    min_separation: float = 0.2,
    score_fn: ScoreFn = score_response,
) -> CalibrationReport:
    """Score each case's graded answers and check ranking quality.

    Returns a CalibrationReport whose ``passed`` is True only when every case is
    ranked monotonically (worst → best never decreases) and the best answer beats
    the worst by at least ``min_separation`` on the 0.0-1.0 scale.
    """
    report = CalibrationReport(min_separation=min_separation)
    for case in cases:
        scores: list[float] = []
        for answer in case.graded_answers:
            result = await score_fn(
                case.prompt,
                answer,
                judge_provider=judge_provider,
                judge_model=judge_model,
                gold_criteria=list(case.gold_criteria) or None,
                max_response_chars=char_budget_for_tokens(512),
            )
            scores.append(result.score)
        separated = (scores[-1] - scores[0]) >= min_separation if scores else False
        report.results.append(
            CaseCalibration(
                case_id=case.id,
                scores=scores,
                monotonic=_is_monotonic(scores),
                separated=separated,
            )
        )
    return report
