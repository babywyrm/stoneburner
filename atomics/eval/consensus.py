"""Combine already-scored judge votes. Does not call a provider."""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class NumericVote:
    score: float
    parse_failed: bool
    judge_model: str
    rationale: str


@dataclass(frozen=True)
class NumericConsensus:
    score: float
    score_stdev: float
    n_judges: int
    parse_failed: bool
    judge_model: str
    rationale: str
    valid_scores: tuple[float, ...]


def combine_numeric(votes: Sequence[NumericVote]) -> NumericConsensus:
    """Mean and population stdev of votes that parsed.

    `votes[0]` is the primary. If nobody parsed, return the primary flagged
    with `n_judges` equal to the panel size.
    """
    if not votes:
        raise ValueError("combine_numeric requires at least one vote")
    primary = votes[0]
    valid = [v for v in votes if not v.parse_failed]
    if not valid:
        return NumericConsensus(
            score=primary.score,
            score_stdev=0.0,
            n_judges=len(votes),
            parse_failed=True,
            judge_model=primary.judge_model,
            rationale=primary.rationale,
            valid_scores=(),
        )
    scores = [v.score for v in valid]
    stdev = round(statistics.pstdev(scores), 3) if len(scores) > 1 else 0.0
    return NumericConsensus(
        score=round(sum(scores) / len(scores), 3),
        score_stdev=stdev,
        n_judges=len(valid),
        parse_failed=False,
        judge_model=", ".join(v.judge_model for v in valid),
        rationale=valid[0].rationale,
        valid_scores=tuple(scores),
    )


@dataclass(frozen=True)
class CategoricalVote:
    label: str
    parse_failed: bool
    judge_model: str
    rationale: str


@dataclass(frozen=True)
class CategoricalConsensus:
    label: str | None
    agreement: float | None
    n_judges: int
    unresolved: bool
    parse_failed: bool
    judge_model: str
    rationale: str
    votes: tuple[str, ...]


def combine_categorical(votes: Sequence[CategoricalVote]) -> CategoricalConsensus:
    """Majority vote. Ties do not invent a winner."""
    if not votes:
        raise ValueError("combine_categorical requires at least one vote")
    primary = votes[0]
    valid = [v for v in votes if not v.parse_failed]
    if not valid:
        return CategoricalConsensus(
            label=None,
            agreement=None,
            n_judges=len(votes),
            unresolved=False,
            parse_failed=True,
            judge_model=primary.judge_model,
            rationale=primary.rationale,
            votes=(),
        )
    counts = Counter(v.label for v in valid)
    top = counts.most_common()
    majority_label, majority_count = top[0]
    tied = len(top) > 1 and top[1][1] == majority_count
    agreement = majority_count / len(valid)
    if tied:
        return CategoricalConsensus(
            label=None,
            agreement=agreement,
            n_judges=len(valid),
            unresolved=True,
            parse_failed=False,
            judge_model=", ".join(v.judge_model for v in valid),
            rationale=valid[0].rationale,
            votes=tuple(v.label for v in valid),
        )
    return CategoricalConsensus(
        label=majority_label,
        agreement=agreement,
        n_judges=len(valid),
        unresolved=False,
        parse_failed=False,
        judge_model=", ".join(v.judge_model for v in valid),
        rationale=valid[0].rationale,
        votes=tuple(v.label for v in valid),
    )
