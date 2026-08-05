"""Integrity accounting for suites that predate the typed attempt contract.

Multi-turn, RAG, red/blue, codegen, and tool-call each grew their own result and
judge types before `AttemptResult` existed. They reported averages computed only
over judges that parsed, and nothing else — so a run where nine of ten judge
calls failed published the tenth score as the headline number, with no signal
that the run was degraded. In an evaluation tool that is the worst failure mode
available: the number looks fine.

These helpers map a suite's own success flags onto the shared vocabulary in
`atomics.eval.outcomes`, so every suite reports the same coverage and
judge-failure figures without rewriting its judges.
"""

from __future__ import annotations

from collections.abc import Sequence

from atomics.eval.outcomes import (
    FixtureOutcome,
    JudgeOutcomeStatus,
    ProviderOutcomeKind,
    RunIntegrity,
)


def fixture_outcome(*, generated: bool, scored: bool) -> FixtureOutcome:
    """Describe one fixture for integrity accounting.

    `generated` is whether the provider produced a response worth judging;
    `scored` is whether a judge returned a usable score for it.

    Generated but not scored is recorded as a judge failure rather than a
    skip. These suites only invoke a judge after a successful generation, so
    the absence of a score there means the judge was asked and gave back
    nothing usable — precisely the case that used to vanish from the averages.
    """
    if not generated:
        return FixtureOutcome(
            generation=ProviderOutcomeKind.PROVIDER_ERROR,
            judge=JudgeOutcomeStatus.SKIPPED,
        )
    return FixtureOutcome(
        generation=ProviderOutcomeKind.COMPLETED,
        judge=(
            JudgeOutcomeStatus.SCORED if scored else JudgeOutcomeStatus.PARSE_FAILED
        ),
    )


def integrity_of(outcomes: Sequence[FixtureOutcome]) -> RunIntegrity:
    """Run integrity for a suite that described its fixtures with `fixture_outcome`."""
    return RunIntegrity.from_fixture_outcomes(outcomes)
