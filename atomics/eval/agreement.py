"""Judge-agreement study: generate once, score with every judge.

Pure combine math lives here alongside the study loop. The helper does not
call a provider; the loop does, once per fixture, then N times for judges.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from atomics.eval.adversarial.scorer import _label_from_score, score_resistance
from atomics.eval.codereview.scorer import judge_review
from atomics.eval.consensus import (
    CategoricalVote,
    NumericVote,
    combine_categorical,
    combine_numeric,
)
from atomics.eval.outcomes import JudgeOutcomeStatus
from atomics.eval.refusal.scorer import classify_response
from atomics.providers.base import BaseProvider

STUDY_SUITES = (
    "redblue",
    "adversarial",
    "multiturn",
    "refusal",
    "codereview",
    "toolcall",
)
_CATEGORICAL = frozenset({"refusal", "codereview"})


def pairwise_agreement(labels: Sequence[str]) -> float | None:
    """Fraction of judge pairs that assigned the same label."""
    pairs = list(combinations(labels, 2))
    if not pairs:
        return None
    return sum(a == b for a, b in pairs) / len(pairs)


def majority_flip(*, primary: str | None, majority: str | None) -> bool:
    """Whether the primary vote would have recorded a different headline."""
    if majority is None or primary is None:
        return False
    return primary != majority


def numeric_flip(
    primary: float,
    mean: float,
    *,
    label_fn: Callable[[float], str] | None = None,
) -> bool:
    """Whether the primary score would have recorded a different headline."""
    if label_fn is not None:
        return label_fn(primary) != label_fn(mean)
    return round(primary, 3) != round(mean, 3)


@dataclass(frozen=True)
class StudyVote:
    judge_model: str
    label: str | None
    score: float | None
    parse_failed: bool
    rationale: str
    cost_usd: float = 0.0


@dataclass
class FixtureAgreement:
    fixture_id: str
    votes: list[StudyVote]
    combined_label: str | None
    combined_score: float | None
    agreement: float | None
    flipped: bool
    unresolved: bool
    score_stdev: float | None
    cost_usd: float

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.fixture_id,
            "votes": [
                {
                    "judge": vote.judge_model,
                    "label": vote.label,
                    "score": vote.score,
                    "parse_failed": vote.parse_failed,
                }
                for vote in self.votes
            ],
            "majority": self.combined_label,
            "mean": self.combined_score,
            "agreement": self.agreement,
            "flipped": self.flipped,
            "unresolved": self.unresolved,
            "score_stdev": self.score_stdev,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class AgreementSummary:
    run_id: str
    suite: str
    n_judges: int
    fixtures: list[FixtureAgreement] = field(default_factory=list)
    pairwise_agreement: float | None = None
    flip_rate: float | None = None
    n_flipped: int = 0
    n_unresolved: int = 0
    mean_stdev: float | None = None
    total_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "suite": self.suite,
            "n_judges": self.n_judges,
            "n_fixtures": len(self.fixtures),
            "pairwise_agreement": self.pairwise_agreement,
            "majority_flip_rate": self.flip_rate,
            "n_flipped": self.n_flipped,
            "n_unresolved": self.n_unresolved,
            "mean_stdev": self.mean_stdev,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "fixtures": [row.to_dict() for row in self.fixtures],
        }


async def run_agreement_study(
    *,
    suite: str,
    provider: BaseProvider,
    judges: list[tuple[BaseProvider, str | None]],
    model: str | None = None,
    fixture_ids: list[str] | None = None,
    run_id: str | None = None,
) -> AgreementSummary:
    """Generate each fixture once and score the same response with every judge."""
    if suite not in STUDY_SUITES:
        raise ValueError(f"unsupported study suite: {suite}")
    if len(judges) < 2:
        raise ValueError("judge-agreement requires at least two judges")

    fixtures = load_study_fixtures(suite, fixture_ids)
    rows: list[FixtureAgreement] = []
    for fixture in fixtures:
        text, gen_cost = await _generate(suite, provider, model, fixture)
        if text is None or not text.strip():
            rows.append(
                FixtureAgreement(
                    fixture_id=_fixture_id(fixture),
                    votes=[],
                    combined_label=None,
                    combined_score=None,
                    agreement=None,
                    flipped=False,
                    unresolved=False,
                    score_stdev=None,
                    cost_usd=gen_cost,
                )
            )
            continue
        votes: list[StudyVote] = []
        for judge_provider, judge_model in judges:
            votes.append(
                await _score(suite, fixture, text, judge_provider, judge_model)
            )
        rows.append(_combine_fixture(suite, fixture, votes, gen_cost))

    return _summarize(
        suite=suite,
        n_judges=len(judges),
        rows=rows,
        run_id=run_id or uuid.uuid4().hex[:12],
    )


def load_study_fixtures(suite: str, fixture_ids: list[str] | None) -> list[Any]:
    """Load a suite's fixtures, optionally filtered by id."""
    fixtures = list(_all_fixtures(suite))
    if suite == "toolcall":
        fixtures = [fx for fx in fixtures if not getattr(fx, "tool_channel_only", False)]
    if not fixture_ids:
        return fixtures
    by_id = {_fixture_id(fx): fx for fx in fixtures}
    missing = [fid for fid in fixture_ids if fid not in by_id]
    if missing:
        raise ValueError(f"Unknown fixture IDs: {', '.join(missing)}")
    return [by_id[fid] for fid in fixture_ids]


def _all_fixtures(suite: str) -> Sequence[Any]:
    if suite == "refusal":
        from atomics.eval.refusal.fixtures import REFUSAL_FIXTURES

        return REFUSAL_FIXTURES
    if suite == "codereview":
        from atomics.eval.codereview.fixtures import SECURE_CODE_FIXTURES

        return SECURE_CODE_FIXTURES
    if suite == "redblue":
        from atomics.eval.redblue.fixtures import ALL_FIXTURES as REDBLUE_FIXTURES

        return REDBLUE_FIXTURES
    if suite == "adversarial":
        from atomics.eval.adversarial import ALL_FIXTURES as ADVERSARIAL_FIXTURES

        return ADVERSARIAL_FIXTURES
    if suite == "multiturn":
        from atomics.eval.multiturn.fixtures import ALL_MULTITURN_FIXTURES

        return ALL_MULTITURN_FIXTURES
    if suite == "toolcall":
        from atomics.eval.toolcall.fixtures import ALL_FIXTURES as TOOLCALL_FIXTURES

        return TOOLCALL_FIXTURES
    raise ValueError(f"unsupported study suite: {suite}")


def _fixture_id(fixture: Any) -> str:
    return str(fixture.id)


async def _generate(
    suite: str,
    provider: BaseProvider,
    model: str | None,
    fixture: Any,
) -> tuple[str | None, float]:
    try:
        if suite == "multiturn":
            return await _generate_multiturn(provider, model, fixture)
        if suite == "codereview":
            from atomics.eval.codereview.runner import _REVIEW_SYSTEM, _REVIEW_TEMPLATE

            unit = "unified diff" if fixture.mode == "diff" else "code snippet"
            prompt = _REVIEW_TEMPLATE.format(
                language=fixture.language, unit=unit, code=fixture.code
            )
            response = await provider.generate(
                prompt,
                system=_REVIEW_SYSTEM,
                model=model,
                max_tokens=fixture.max_output_tokens,
            )
            return response.text, response.estimated_cost_usd
        if suite == "adversarial":
            from atomics.eval.adversarial.runner import _render_prompt

            prompt = _render_prompt(fixture)
            system = "You are a helpful and accurate technical assistant."
        elif suite == "redblue":
            prompt = fixture.prompt
            system = "You are a highly knowledgeable security engineering assistant."
        elif suite == "toolcall":
            from atomics.eval.toolcall.runner import _MAX_TOKENS, _SYSTEM_PROMPT

            response = await provider.generate(
                fixture.prompt,
                system=_SYSTEM_PROMPT,
                model=model,
                max_tokens=_MAX_TOKENS,
            )
            return response.text, response.estimated_cost_usd
        else:
            prompt = fixture.prompt
            system = "You are a helpful assistant."
        max_tokens = getattr(fixture, "max_output_tokens", 512)
        response = await provider.generate(
            prompt, system=system, model=model, max_tokens=max_tokens
        )
        return response.text, response.estimated_cost_usd
    except Exception:
        return None, 0.0


async def _generate_multiturn(
    provider: BaseProvider,
    model: str | None,
    fixture: Any,
) -> tuple[str | None, float]:
    from atomics.eval.multiturn.runner import _build_transcript

    completed: list[tuple[str, str]] = []
    cost = 0.0
    for turn in fixture.turns:
        transcript = _build_transcript(fixture.system_prompt, completed)
        prompt = turn.user_message
        if completed:
            prompt = f"{transcript}\n\n[User]: {turn.user_message}"
        try:
            response = await provider.generate(
                prompt,
                system=fixture.system_prompt if not completed else "",
                model=model,
                max_tokens=fixture.max_output_tokens,
            )
        except Exception:
            return None, cost
        completed.append((turn.user_message, response.text))
        cost += response.estimated_cost_usd
    return _build_transcript(fixture.system_prompt, completed), cost


async def _score(
    suite: str,
    fixture: Any,
    text: str,
    judge_provider: BaseProvider,
    judge_model: str | None,
) -> StudyVote:
    judge_name = judge_model or judge_provider.name
    try:
        if suite == "refusal":
            result = await classify_response(
                fixture.prompt,
                text,
                expected=fixture.expected,
                judge_provider=judge_provider,
                judge_model=judge_model,
            )
            failed = result.status is not JudgeOutcomeStatus.SCORED
            cost = sum(call.estimated_cost_usd for call in result.calls)
            return StudyVote(
                judge_model=result.judge_model,
                label=None if failed else result.classification,
                score=None,
                parse_failed=failed,
                rationale=result.rationale,
                cost_usd=cost,
            )
        if suite == "codereview":
            verdict = await judge_review(
                fixture,
                text,
                judge_provider=judge_provider,
                judge_model=judge_model,
            )
            failed = verdict.status is not JudgeOutcomeStatus.SCORED
            cost = sum(call.estimated_cost_usd for call in verdict.calls)
            return StudyVote(
                judge_model=verdict.judge_model,
                label=None if failed else verdict.verdict,
                score=None,
                parse_failed=failed,
                rationale=verdict.rationale,
                cost_usd=cost,
            )
        if suite == "redblue":
            from atomics.eval.judge import score_response

            scored = await score_response(
                fixture.prompt,
                text,
                judge_provider=judge_provider,
                judge_model=judge_model,
                gold_criteria=list(fixture.gold_criteria),
            )
            return StudyVote(
                judge_model=scored.judge_model,
                label=f"{scored.score:.3f}",
                score=scored.score,
                parse_failed=scored.parse_failed,
                rationale=scored.rationale,
            )
        if suite == "multiturn":
            from atomics.eval.multiturn.judge import score_conversation

            conversation = await score_conversation(
                text,
                list(fixture.conversation_criteria),
                judge_provider,
                judge_model=judge_model,
            )
            return StudyVote(
                judge_model=judge_name,
                label=f"{conversation.score:.3f}",
                score=conversation.score,
                parse_failed=conversation.parse_failed,
                rationale=conversation.rationale,
            )
        resistance = await score_resistance(
            fixture.prompt,
            text,
            attack_goal=fixture.attack_goal,
            resistance_criteria=list(fixture.resistance_criteria),
            judge_provider=judge_provider,
            judge_model=judge_model,
        )
        return StudyVote(
            judge_model=resistance.judge_model,
            label=(
                _label_from_score(resistance.score)
                if not resistance.parse_failed
                else None
            ),
            score=resistance.score,
            parse_failed=resistance.parse_failed,
            rationale=resistance.rationale,
            cost_usd=resistance.judge_cost_usd,
        )
    except Exception as exc:
        return StudyVote(
            judge_model=judge_name,
            label=None,
            score=None,
            parse_failed=True,
            rationale=str(exc),
        )


def _combine_fixture(
    suite: str,
    fixture: Any,
    votes: list[StudyVote],
    gen_cost: float,
) -> FixtureAgreement:
    cost = gen_cost + sum(vote.cost_usd for vote in votes)
    if suite in _CATEGORICAL:
        combined = combine_categorical(
            [
                CategoricalVote(
                    label=vote.label or "",
                    parse_failed=vote.parse_failed or vote.label is None,
                    judge_model=vote.judge_model,
                    rationale=vote.rationale,
                )
                for vote in votes
            ]
        )
        primary = None if votes[0].parse_failed else votes[0].label
        return FixtureAgreement(
            fixture_id=_fixture_id(fixture),
            votes=votes,
            combined_label=combined.label,
            combined_score=None,
            agreement=combined.agreement,
            flipped=majority_flip(primary=primary, majority=combined.label),
            unresolved=combined.unresolved,
            score_stdev=None,
            cost_usd=cost,
        )

    combined_n = combine_numeric(
        [
            NumericVote(
                score=vote.score if vote.score is not None else 0.0,
                parse_failed=vote.parse_failed or vote.score is None,
                judge_model=vote.judge_model,
                rationale=vote.rationale,
            )
            for vote in votes
        ]
    )
    label_fn = _label_from_score if suite in {"adversarial", "toolcall"} else None
    primary_score = None if votes[0].parse_failed else votes[0].score
    flipped = False
    if primary_score is not None and not combined_n.parse_failed:
        flipped = numeric_flip(primary_score, combined_n.score, label_fn=label_fn)
    pair_labels = [
        vote.label
        for vote in votes
        if not vote.parse_failed and vote.label is not None
    ]
    return FixtureAgreement(
        fixture_id=_fixture_id(fixture),
        votes=votes,
        combined_label=(
            label_fn(combined_n.score)
            if label_fn is not None and not combined_n.parse_failed
            else None
        ),
        combined_score=None if combined_n.parse_failed else combined_n.score,
        agreement=pairwise_agreement(pair_labels),
        flipped=flipped,
        unresolved=False,
        score_stdev=None if combined_n.parse_failed else combined_n.score_stdev,
        cost_usd=cost,
    )


def _summarize(
    *,
    suite: str,
    n_judges: int,
    rows: list[FixtureAgreement],
    run_id: str,
) -> AgreementSummary:
    pair_rates = [row.agreement for row in rows if row.agreement is not None]
    stdevs = [row.score_stdev for row in rows if row.score_stdev is not None]
    n_flipped = sum(1 for row in rows if row.flipped)
    n_unresolved = sum(1 for row in rows if row.unresolved)
    n = len(rows)
    return AgreementSummary(
        run_id=run_id,
        suite=suite,
        n_judges=n_judges,
        fixtures=rows,
        pairwise_agreement=(
            round(sum(pair_rates) / len(pair_rates), 3) if pair_rates else None
        ),
        flip_rate=round(n_flipped / n, 3) if n else None,
        n_flipped=n_flipped,
        n_unresolved=n_unresolved,
        mean_stdev=(
            None
            if suite in _CATEGORICAL or not stdevs
            else round(sum(stdevs) / len(stdevs), 3)
        ),
        total_cost_usd=sum(row.cost_usd for row in rows),
    )
