"""Integrity accounting for suites that predate the typed attempt contract.

The motivating bug: multi-turn, RAG, and red/blue averaged only over judges
that parsed, and reported nothing else. A run where nine of ten judge calls
failed published the tenth score as the headline number and looked healthy. In
an evaluation tool that is the worst available failure mode, because the number
looks fine.
"""

from __future__ import annotations

import pytest

from atomics.eval.outcomes import (
    FixtureOutcome,
    JudgeOutcomeStatus,
    ProviderOutcomeKind,
    RunIntegrity,
    RunStatus,
)
from atomics.eval.suite_integrity import (
    fixture_outcome,
    format_headline_rate,
    headline_rate,
    integrity_of,
)


def outcomes(*pairs: tuple[bool, bool]) -> list[FixtureOutcome]:
    """Build fixture outcomes from (generated, scored) pairs."""
    return [fixture_outcome(generated=g, scored=s) for g, s in pairs]


class TestHeadlineRate:
    """A scored-subset average is not a finished score."""

    def test_complete_run_keeps_the_number(self):
        integrity = integrity_of(outcomes((True, True), (True, True)))
        assert headline_rate(0.973, integrity) == 0.973
        assert format_headline_rate(0.973, integrity) == "97.3%"

    def test_partial_run_refuses_a_bare_percentage(self):
        integrity = integrity_of(outcomes((True, True), (True, False)))
        assert headline_rate(1.0, integrity) is None
        assert format_headline_rate(1.0, integrity) == "n/a (1/2 scored)"

    def test_unscored_run_names_the_empty_denominator(self):
        integrity = integrity_of(outcomes((True, False), (True, False)))
        assert headline_rate(None, integrity) is None
        assert format_headline_rate(None, integrity) == "n/a (0/2 scored)"

    def test_complete_run_with_no_score_is_plain_na(self):
        integrity = integrity_of(outcomes((True, True)))
        assert format_headline_rate(None, integrity) == "n/a"


class TestFixtureOutcomeMapping:
    def test_a_scored_fixture_is_complete(self):
        outcome = fixture_outcome(generated=True, scored=True)
        assert outcome.generation is ProviderOutcomeKind.COMPLETED
        assert outcome.judge is JudgeOutcomeStatus.SCORED

    def test_generated_but_unscored_is_a_judge_failure_not_a_skip(self):
        """These suites only judge after a successful generation, so a missing
        score there means the judge was asked and returned nothing usable."""
        outcome = fixture_outcome(generated=True, scored=False)
        assert outcome.generation is ProviderOutcomeKind.COMPLETED
        assert outcome.judge is JudgeOutcomeStatus.PARSE_FAILED

    def test_a_failed_generation_skips_the_judge(self):
        """Not a judge failure: nothing was ever asked of it."""
        outcome = fixture_outcome(generated=False, scored=False)
        assert outcome.generation is ProviderOutcomeKind.PROVIDER_ERROR
        assert outcome.judge is JudgeOutcomeStatus.SKIPPED


class TestIntegrityCounting:
    def test_a_clean_run_is_complete(self):
        integrity = integrity_of(outcomes((True, True), (True, True)))

        assert integrity.status is RunStatus.COMPLETE
        assert integrity.fixture_coverage == 1.0
        assert integrity.judge_failure_rate == 0.0
        assert integrity.should_exit_nonzero is False

    def test_the_motivating_case_one_score_out_of_ten(self):
        """Nine judge failures used to be invisible; the tenth score was the
        headline number and nothing said the run was degraded."""
        integrity = integrity_of(outcomes(*([(True, True)] + [(True, False)] * 9)))

        assert integrity.status is RunStatus.PARTIAL
        assert integrity.fixtures_scored == 1
        assert integrity.fixtures_total == 10
        assert integrity.fixture_coverage == 0.1
        assert integrity.judge_failures == 9
        assert integrity.judge_failure_rate == 0.9
        assert integrity.should_exit_nonzero is True

    def test_a_total_judge_outage_is_infrastructure_invalid(self):
        integrity = integrity_of(outcomes(*([(True, False)] * 5)))

        assert integrity.status is RunStatus.INFRASTRUCTURE_INVALID
        assert integrity.fixtures_scored == 0
        assert integrity.should_exit_nonzero is True

    def test_generation_failures_are_counted_apart_from_judge_failures(self):
        """A broken provider and a broken judge are different problems."""
        integrity = integrity_of(outcomes((False, False), (True, True)))

        assert integrity.generation_failures == 1
        assert integrity.judge_failures == 0
        assert integrity.attempts_scorable == 1

    def test_a_failed_generation_is_not_blamed_on_the_judge(self):
        integrity = integrity_of(outcomes(*([(False, False)] * 4)))

        assert integrity.generation_failures == 4
        assert integrity.judge_failures == 0
        assert integrity.judge_failure_rate == 0.0
        assert integrity.status is RunStatus.INFRASTRUCTURE_INVALID

    def test_an_empty_run_does_not_divide_by_zero(self):
        integrity = integrity_of([])

        assert integrity.fixtures_total == 0
        assert integrity.fixture_coverage == 0.0
        assert integrity.judge_failure_rate == 0.0
        assert integrity.status is RunStatus.INFRASTRUCTURE_INVALID

    def test_one_fixture_is_one_attempt(self):
        integrity = integrity_of(outcomes((True, True), (True, False)))

        assert integrity.attempts_total == integrity.fixtures_total == 2


class TestBothConstructorsAgree:
    """The two paths must never describe the same run differently."""

    def test_the_same_run_counts_the_same_either_way(self):
        from atomics.eval.outcomes import (
            AttemptResult,
            JudgeOutcome,
            ProviderOutcome,
        )

        def attempt(index: int, *, generated: bool, scored: bool) -> AttemptResult:
            judge: JudgeOutcome | None
            if not generated:
                judge = JudgeOutcome(status=JudgeOutcomeStatus.SKIPPED)
            elif scored:
                judge = JudgeOutcome(
                    status=JudgeOutcomeStatus.SCORED,
                    score=0.5,
                    judges_expected=1,
                    judges_scored=1,
                )
            else:
                judge = JudgeOutcome(status=JudgeOutcomeStatus.PARSE_FAILED)
            return AttemptResult(
                attempt_index=index,
                provider=ProviderOutcome(
                    kind=(
                        ProviderOutcomeKind.COMPLETED
                        if generated
                        else ProviderOutcomeKind.PROVIDER_ERROR
                    )
                ),
                response_text="x" if generated else "",
                latency_ms=1.0,
                estimated_cost_usd=0.0,
                input_tokens=0,
                output_tokens=0,
                thinking_tokens=0,
                judge=judge,
            )

        shape = [(True, True), (True, False), (False, False), (True, True)]

        from_attempts = RunIntegrity.from_fixture_attempts(
            [[attempt(i, generated=g, scored=s)] for i, (g, s) in enumerate(shape)]
        )
        from_outcomes = integrity_of(outcomes(*shape))

        assert from_attempts == from_outcomes


class TestSuitesReportIntegrity:
    """Every suite must expose the same integrity block in its JSON."""

    @pytest.mark.parametrize(
        "summary_factory",
        [
            pytest.param("multiturn", id="multiturn"),
            pytest.param("rag", id="rag"),
            pytest.param("redblue", id="redblue"),
            pytest.param("codegen", id="codegen"),
            pytest.param("toolcall", id="toolcall"),
        ],
    )
    def test_the_json_carries_an_integrity_block(self, summary_factory):
        summary = _empty_summary(summary_factory)
        payload = summary.to_dict()

        assert "integrity" in payload, f"{summary_factory} omits integrity"
        block = payload["integrity"]
        for key in (
            "status",
            "fixtures_total",
            "fixtures_scored",
            "fixture_coverage",
            "judge_failure_rate",
            "should_exit_nonzero",
        ):
            assert key in block, f"{summary_factory} integrity missing {key}"


class TestDegradedRunsAreVisible:
    """The headline score stays as it was; integrity is what exposes the rot.

    These build real summary objects rather than outcome pairs, so they fail if
    a suite's wiring stops matching the shape of its own results.
    """

    def test_multiturn_reports_a_confident_score_over_one_conversation(self):
        from atomics.eval.multiturn.runner import MultiturnRunSummary

        summary = _empty_summary("multiturn")
        assert isinstance(summary, MultiturnRunSummary)
        summary.conversation_results = [
            _conversation(scored=i == 0) for i in range(10)
        ]

        # Unchanged behavior: the average is still computed over the one
        # conversation that scored, and still looks like a normal result.
        assert summary.avg_turn_score == 0.9

        integrity = summary.integrity
        assert integrity.fixtures_scored == 1
        assert integrity.fixtures_total == 10
        assert integrity.fixture_coverage == 0.1
        assert integrity.judge_failure_rate == 0.9
        assert integrity.should_exit_nonzero is True
        assert summary.to_dict()["integrity"]["status"] == "partial"

    def test_rag_reports_the_same_shape(self):
        summary = _empty_summary("rag")
        summary.fixture_results = [_rag_result(scored=i == 0) for i in range(4)]

        assert summary.overall_rag_score is not None
        assert summary.integrity.fixture_coverage == 0.25
        assert summary.integrity.judge_failure_rate == 0.75

    def test_rag_parse_failure_rate_still_answers(self):
        """It was the only integrity signal any of these suites had."""
        summary = _empty_summary("rag")
        summary.fixture_results = [_rag_result(scored=i == 0) for i in range(4)]

        assert summary.parse_failure_rate == 0.75
        assert summary.to_dict()["parse_failure_rate"] == 0.75

    def test_multiturn_counts_a_failed_conversation_judge(self):
        """Found by running a real suite, not by any unit test.

        `overall_score` falls back to the turn scores when the conversation
        judge fails, so the first version of this reported `complete` with zero
        judge failures on a run whose retention and consistency were empty.
        """
        summary = _empty_summary("multiturn")
        summary.conversation_results = [
            _conversation(scored=True, conversation_scored=False) for _ in range(4)
        ]

        assert summary.avg_turn_score == 0.9
        assert summary.avg_retention is None
        assert summary.avg_consistency is None

        assert summary.integrity.status.value == "infrastructure_invalid"
        assert summary.integrity.judge_failures == 4
        assert summary.integrity.should_exit_nonzero is True

    def test_a_clean_multiturn_run_is_complete(self):
        summary = _empty_summary("multiturn")
        summary.conversation_results = [_conversation(scored=True) for _ in range(3)]

        assert summary.integrity.status.value == "complete"
        assert summary.integrity.should_exit_nonzero is False

    def test_codegen_distinguishes_a_broken_run_from_a_bad_model(self):
        """Both report overall_pass_rate 0.0; only integrity tells them apart."""
        broken = _empty_summary("codegen")
        broken.fixture_results = [_codegen_result(generated=False) for _ in range(3)]

        bad_model = _empty_summary("codegen")
        bad_model.fixture_results = [
            _codegen_result(generated=True, passed=0) for _ in range(3)
        ]

        assert broken.overall_pass_rate == bad_model.overall_pass_rate == 0.0

        assert broken.integrity.generation_failures == 3
        assert broken.integrity.should_exit_nonzero is True
        assert bad_model.integrity.generation_failures == 0
        assert bad_model.integrity.should_exit_nonzero is False


def _task_result(*, failed: bool = False):
    from datetime import UTC, datetime

    from atomics.models import TaskCategory, TaskResult, TaskStatus

    now = datetime.now(UTC)
    return TaskResult(
        task_id="t1",
        run_id="r1",
        category=TaskCategory.GENERAL_QA,
        task_name="fx",
        provider="ollama",
        model="m",
        status=TaskStatus.FAILED if failed else TaskStatus.SUCCESS,
        started_at=now,
        completed_at=now,
    )


def _conversation(*, scored: bool, conversation_scored: bool | None = None):
    """One conversation result.

    `conversation_scored` defaults to `scored` and can be set independently to
    model a run where the turn judges worked and the conversation judge did not.
    """
    from atomics.eval.multiturn.fixtures import ALL_MULTITURN_FIXTURES
    from atomics.eval.multiturn.judge import ConversationJudgeResult, TurnJudgeResult
    from atomics.eval.multiturn.runner import ConversationResult, TurnResult

    if conversation_scored is None:
        conversation_scored = scored

    turn = TurnResult(
        turn_index=0,
        user_message="hi",
        response="hello",
        latency_ms=1.0,
        tokens=1,
        cost=0.0,
        judge=TurnJudgeResult(
            accuracy=4,
            context_use=4,
            coherence=4,
            score=0.9,
            rationale="",
            parse_failed=not scored,
        ),
    )
    return ConversationResult(
        fixture=ALL_MULTITURN_FIXTURES[0],
        turn_results=[turn],
        conversation_judge=ConversationJudgeResult(
            retention=4,
            consistency=3,
            instruction=3,
            score=0.9,
            rationale="",
            parse_failed=not conversation_scored,
        ),
        task_result=_task_result(),
        overall_score=0.9 if scored else None,
    )


def _rag_result(*, scored: bool):
    from atomics.eval.rag.fixtures import ALL_RAG_FIXTURES
    from atomics.eval.rag.judge import RAGJudgeResult
    from atomics.eval.rag.runner import RAGFixtureResult

    return RAGFixtureResult(
        fixture=ALL_RAG_FIXTURES[0],
        task_result=_task_result(),
        judge=RAGJudgeResult(
            grounding=4,
            faithfulness=3,
            abstention=2,
            score=0.8,
            rationale="",
            parse_failed=not scored,
        ),
    )


def _codegen_result(*, generated: bool, passed: int = 0):
    from atomics.eval.codegen.fixtures import ALL_CODEGEN_FIXTURES
    from atomics.eval.codegen.runner import CodegenFixtureResult

    return CodegenFixtureResult(
        fixture=ALL_CODEGEN_FIXTURES[0],
        task_result=_task_result(failed=not generated),
        tests_passed=passed,
        tests_total=2,
        pass_rate=passed / 2,
        extracted_code=None,
        test_details=[],
    )


def _empty_summary(suite: str):
    """A summary with no fixture results, which every suite can build."""
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    common = dict(
        run_id="r1", provider="ollama", model="m", started_at=now, completed_at=now
    )

    if suite == "multiturn":
        from atomics.eval.multiturn.runner import MultiturnRunSummary

        return MultiturnRunSummary(judge_provider="ollama", judge_model="j", **common)
    if suite == "rag":
        from atomics.eval.rag.runner import RAGRunSummary

        return RAGRunSummary(judge_provider="ollama", judge_model="j", **common)
    if suite == "redblue":
        from atomics.eval.redblue.runner import RedBlueSummary

        return RedBlueSummary(mode="both", **common)
    if suite == "codegen":
        from atomics.eval.codegen.runner import CodegenRunSummary

        return CodegenRunSummary(**common)

    from atomics.eval.toolcall.runner import ToolCallSummary

    return ToolCallSummary(
        run_id="r1",
        provider="ollama",
        model="m",
        started_at=now.isoformat(),
        completed_at=now.isoformat(),
        tool_capable=True,
    )
