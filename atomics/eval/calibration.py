"""Judge calibration: verify the quality judge ranks graded answers correctly.

A judge is only trustworthy if a clearly-better answer reliably scores higher
than a clearly-worse one. This module ships a small set of fixtures, each with
several answers ordered worst → best, and a harness that scores them and checks
that the resulting scores are monotonically non-decreasing and well separated.

Used as a regression guard (see tests/test_calibration.py): a judge-prompt or
parsing change that breaks ranking will fail the calibration check.

`run_judge_calibration` is the absolute check: a human-graded set scored on the
judge's own rubric, reported as mean error and bias against the human.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from atomics.eval.judge import JudgeResult, char_budget_for_tokens, score_response
from atomics.eval.outcomes import RunIntegrity, RunStatus
from atomics.eval.suite_integrity import fixture_outcome, integrity_of
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
    monotonic: bool  # scores never decrease from worst → best
    separated: bool  # best - worst >= min_separation


@dataclass
class CalibrationReport:
    results: list[CaseCalibration] = field(default_factory=list)
    min_separation: float = 0.2

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(r.monotonic and r.separated for r in self.results)

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


CALIBRATE_SUITE = "judge-calibrate"
DEFAULT_MAX_ERROR = 0.15
_GRADE_RANGES = {"accuracy": 4, "completeness": 3, "format": 3}

Pair = tuple[float, float]  # (human, judge), both 0.0-1.0


def mean_abs_error(pairs: Sequence[Pair]) -> float | None:
    if not pairs:
        return None
    return round(sum(abs(judge - human) for human, judge in pairs) / len(pairs), 3)


def mean_bias(pairs: Sequence[Pair]) -> float | None:
    """Mean of judge minus human: positive means the judge scores high."""
    if not pairs:
        return None
    return round(sum(judge - human for human, judge in pairs) / len(pairs), 3)


def pair_order_agreement(pairs: Sequence[Pair]) -> float | None:
    """Share of pairs the human ranked apart that the judge orders the same way."""
    ranked = [(a, b) for a, b in itertools.combinations(pairs, 2) if a[0] != b[0]]
    if not ranked:
        return None
    same = sum((a[0] - b[0]) * (a[1] - b[1]) > 0 for a, b in ranked)
    return round(same / len(ranked), 3)


@dataclass(frozen=True)
class GradedItem:
    id: str
    task: str
    answer: str
    human: float  # (accuracy + completeness + format) / 10, as eval/judge.py
    gold_criteria: tuple[str, ...] = ()


@dataclass(frozen=True)
class GradedSet:
    name: str
    digest: str
    items: tuple[GradedItem, ...]


def default_set_path() -> Path:
    return Path(__file__).parent / "calibration_sets" / "redblue-blue.yaml"


def load_graded_set(path: Path) -> GradedSet:
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    entries = data.get("items") if isinstance(data, dict) else None
    if not isinstance(data, dict) or not isinstance(entries, list) or not entries:
        raise ValueError(f"{path.name}: needs a non-empty items list")
    items: list[GradedItem] = []
    for index, entry in enumerate(entries, start=1):
        item = _graded_item(entry, where=f"{path.name} item {index}")
        if any(seen.id == item.id for seen in items):
            raise ValueError(f"{path.name}: duplicate id {item.id}")
        items.append(item)
    return GradedSet(
        name=str(data.get("name") or path.stem),
        digest=hashlib.sha256(raw).hexdigest()[:12],
        items=tuple(items),
    )


def _graded_item(entry: object, *, where: str) -> GradedItem:
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: not a mapping")
    for key in ("id", "task", "answer", "human"):
        if not entry.get(key):
            raise ValueError(f"{where}: missing {key}")
    human = entry["human"]
    if not isinstance(human, dict):
        raise ValueError(f"{where}: human must be a mapping")
    total = 0
    for name, top in _GRADE_RANGES.items():
        value = human.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= top:
            raise ValueError(f"{where}: human.{name} must be an integer 0-{top}")
        total += value
    gold = entry.get("gold_criteria") or []
    if not isinstance(gold, list) or not all(isinstance(g, str) for g in gold):
        raise ValueError(f"{where}: gold_criteria must be a list of strings")
    return GradedItem(
        id=str(entry["id"]),
        task=str(entry["task"]),
        answer=str(entry["answer"]),
        human=round(total / 10, 3),
        gold_criteria=tuple(gold),
    )


@dataclass(frozen=True)
class ItemCalibration:
    id: str
    human: float
    judge: float | None  # None: the judge failed or its reply did not parse


@dataclass
class JudgeCalibration:
    set_name: str
    set_digest: str
    judge_provider: str
    judge_model: str
    max_error: float
    items: list[ItemCalibration]

    @property
    def pairs(self) -> list[Pair]:
        return [(i.human, i.judge) for i in self.items if i.judge is not None]

    @property
    def integrity(self) -> RunIntegrity:
        return integrity_of(
            [fixture_outcome(generated=True, scored=i.judge is not None) for i in self.items]
        )

    @property
    def mae(self) -> float | None:
        return mean_abs_error(self.pairs)

    @property
    def bias(self) -> float | None:
        return mean_bias(self.pairs)

    @property
    def pair_order(self) -> float | None:
        return pair_order_agreement(self.pairs)

    @property
    def passed(self) -> bool:
        mae = self.mae
        complete = self.integrity.status is RunStatus.COMPLETE
        return complete and mae is not None and mae <= self.max_error

    def to_dict(self) -> dict[str, object]:
        return {
            "set_name": self.set_name,
            "set_digest": self.set_digest,
            "judge_provider": self.judge_provider,
            "judge_model": self.judge_model,
            "max_error": self.max_error,
            "mae": self.mae,
            "bias": self.bias,
            "pair_order": self.pair_order,
            "passed": self.passed,
            "integrity": self.integrity.status.value,
            "items": [asdict(i) for i in self.items],
        }


async def run_judge_calibration(
    judge_provider: BaseProvider,
    *,
    provider_name: str,
    judge_model: str | None,
    graded: GradedSet,
    max_error: float = DEFAULT_MAX_ERROR,
    score_fn: ScoreFn = score_response,
    on_item: Callable[[ItemCalibration], None] | None = None,
) -> JudgeCalibration:
    """Score every graded answer with one judge. Stored answers are judged untruncated."""
    items: list[ItemCalibration] = []
    for item in graded.items:
        result = await score_fn(
            item.task,
            item.answer,
            judge_provider=judge_provider,
            judge_model=judge_model,
            gold_criteria=list(item.gold_criteria) or None,
            max_response_chars=len(item.answer) + 1,
        )
        row = ItemCalibration(item.id, item.human, None if result.parse_failed else result.score)
        items.append(row)
        if on_item is not None:
            on_item(row)
    return JudgeCalibration(
        set_name=graded.name,
        set_digest=graded.digest,
        judge_provider=provider_name,
        judge_model=judge_model or judge_provider.default_model or "default",
        max_error=max_error,
        items=items,
    )
