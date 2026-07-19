"""Tests for the multi-turn conversation evaluation suite."""

from __future__ import annotations

import pytest

from atomics.eval.multiturn import ConversationFixture, ConversationTurn
from atomics.eval.multiturn.fixtures import ALL_MULTITURN_FIXTURES
from atomics.eval.multiturn.judge import (
    TurnJudgeResult,
    ConversationJudgeResult,
    _parse_turn_rubric,
    _parse_conv_rubric,
)
from atomics.eval.multiturn.runner import (
    ConversationResult,
    MultiturnRunSummary,
    TurnResult,
    _build_transcript,
)
from atomics.models import TaskCategory, TaskComplexity, TaskResult, TaskStatus


# ── Model tests ──────────────────────────────────────────────────────────────


def test_conversation_turn_creation():
    turn = ConversationTurn(
        user_message="Hello",
        expected_behavior="Greet back",
        gold_criteria=["greeting"],
    )
    assert turn.user_message == "Hello"


def test_conversation_fixture_defaults():
    f = ConversationFixture(
        id="test-1",
        complexity=TaskComplexity.LIGHT,
        system_prompt="You are helpful.",
        turns=[ConversationTurn("Hi", "Greet", ["hello"])],
    )
    assert f.max_output_tokens == 512
    assert f.conversation_criteria == []


# ── Fixtures collection tests ────────────────────────────────────────────────


def test_all_fixtures_loaded():
    assert len(ALL_MULTITURN_FIXTURES) == 35


def test_fixture_ids_are_unique():
    ids = [f.id for f in ALL_MULTITURN_FIXTURES]
    assert len(ids) == len(set(ids))


def test_fixture_ids_follow_convention():
    for f in ALL_MULTITURN_FIXTURES:
        assert f.id.startswith("mt-eval-"), f"Fixture {f.id} doesn't follow mt-eval-NN pattern"


def test_fixtures_have_multiple_turns():
    for f in ALL_MULTITURN_FIXTURES:
        assert len(f.turns) >= 2, f"Fixture {f.id} has fewer than 2 turns"


def test_fixtures_have_conversation_criteria():
    for f in ALL_MULTITURN_FIXTURES:
        assert len(f.conversation_criteria) >= 1, f"Fixture {f.id} has no conversation criteria"


def test_complexity_spread():
    complexities = {f.complexity for f in ALL_MULTITURN_FIXTURES}
    assert TaskComplexity.LIGHT in complexities
    assert TaskComplexity.MODERATE in complexities
    assert TaskComplexity.HEAVY in complexities


def test_turns_have_expected_behavior():
    for f in ALL_MULTITURN_FIXTURES:
        for i, t in enumerate(f.turns):
            assert t.expected_behavior, f"Fixture {f.id} turn {i} has no expected_behavior"


def _ids_in_range(prefix: str, start: int, end: int) -> set[str]:
    return {f"{prefix}-{i:02d}" for i in range(start, end + 1)}


def test_contradiction_fixtures_present():
    expected = _ids_in_range("mt-eval", 16, 20)
    found = {f.id for f in ALL_MULTITURN_FIXTURES if f.id in expected}
    assert found, "No contradiction detection fixtures found"
    assert found == expected, f"Missing contradiction fixtures: {expected - found}"


def test_persona_fixtures_present():
    expected = _ids_in_range("mt-eval", 21, 24)
    found = {f.id for f in ALL_MULTITURN_FIXTURES if f.id in expected}
    assert found, "No persona drift fixtures found"
    assert found == expected, f"Missing persona fixtures: {expected - found}"


def test_long_context_fixtures_present():
    expected = _ids_in_range("mt-eval", 25, 28)
    found = {f.id for f in ALL_MULTITURN_FIXTURES if f.id in expected}
    assert found, "No long-context retention fixtures found"
    assert found == expected, f"Missing long-context fixtures: {expected - found}"


def test_tool_use_fixtures_present():
    expected = _ids_in_range("mt-eval", 29, 32)
    found = {f.id for f in ALL_MULTITURN_FIXTURES if f.id in expected}
    assert found, "No multi-turn tool-use fixtures found"
    assert found == expected, f"Missing tool-use fixtures: {expected - found}"


def test_security_multiturn_fixtures_present():
    expected = _ids_in_range("mt-eval", 33, 35)
    found = {f.id for f in ALL_MULTITURN_FIXTURES if f.id in expected}
    assert found, "No security-focused multi-turn fixtures found"
    assert found == expected, f"Missing security multi-turn fixtures: {expected - found}"


def test_long_context_fixtures_have_min_turns():
    long_context_ids = _ids_in_range("mt-eval", 25, 28)
    long_fixtures = [f for f in ALL_MULTITURN_FIXTURES if f.id in long_context_ids]
    assert any(len(f.turns) >= 8 for f in long_fixtures), (
        "No long-context fixture has 8+ turns"
    )


# ── Turn judge parse tests ───────────────────────────────────────────────────


def test_parse_turn_strict():
    raw = "ACCURACY: 4\nCONTEXT_USE: 3\nCOHERENCE: 2\nRATIONALE: Good context use."
    result = _parse_turn_rubric(raw)
    assert result is not None
    assert result == (4, 3, 2, "Good context use.")


def test_parse_turn_lenient():
    raw = "**Accuracy** - 3\nContext Use: 2\nCoherence: 1\nRationale: OK response."
    result = _parse_turn_rubric(raw)
    assert result is not None
    assert result[0] == 3
    assert result[1] == 2


def test_parse_turn_clamped():
    raw = "ACCURACY: 9\nCONTEXT_USE: 7\nCOHERENCE: 5\nRATIONALE: Over."
    result = _parse_turn_rubric(raw)
    assert result is not None
    assert result[0] == 4
    assert result[1] == 3
    assert result[2] == 3


def test_parse_turn_garbage():
    assert _parse_turn_rubric("random text") is None


# ── Conversation judge parse tests ───────────────────────────────────────────


def test_parse_conv_strict():
    raw = "RETENTION: 4\nCONSISTENCY: 3\nINSTRUCTION: 2\nRATIONALE: Strong retention."
    result = _parse_conv_rubric(raw)
    assert result is not None
    assert result == (4, 3, 2, "Strong retention.")


def test_parse_conv_lenient():
    raw = "retention: 3\nconsistency: 2\ninstruction: 1\nrationale: decent."
    result = _parse_conv_rubric(raw)
    assert result is not None
    assert result[0] == 3


def test_parse_conv_garbage():
    assert _parse_conv_rubric("gibberish") is None


# ── Transcript builder tests ─────────────────────────────────────────────────


def test_build_transcript_empty():
    t = _build_transcript("Be helpful.", [])
    assert "[System]: Be helpful." in t


def test_build_transcript_with_turns():
    turns = [("Hello", "Hi!"), ("What is 2+2?", "4")]
    t = _build_transcript("System", turns)
    assert "[User]: Hello" in t
    assert "[Assistant]: Hi!" in t
    assert "[User]: What is 2+2?" in t
    assert "[Assistant]: 4" in t


# ── Summary tests ────────────────────────────────────────────────────────────


def _make_conversation_result(
    fixture_id: str = "mt-eval-01",
    turn_scores: list[float] | None = None,
    conv_score: float | None = None,
) -> ConversationResult:
    from datetime import UTC, datetime

    fixture = ConversationFixture(
        id=fixture_id,
        complexity=TaskComplexity.LIGHT,
        system_prompt="test",
        turns=[ConversationTurn("q", "a", [])],
    )

    turn_results = []
    if turn_scores:
        for i, s in enumerate(turn_scores):
            turn_results.append(TurnResult(
                turn_index=i, user_message="q", response="a",
                latency_ms=100.0, tokens=50, cost=0.001,
                judge=TurnJudgeResult(3, 2, 2, s, "ok"),
            ))

    conv_judge = None
    if conv_score is not None:
        conv_judge = ConversationJudgeResult(3, 2, 2, conv_score, "good")

    overall = None
    if turn_scores and conv_score is not None:
        avg_turn = sum(turn_scores) / len(turn_scores)
        overall = round((avg_turn + conv_score) / 2, 3)

    tr = TaskResult(
        run_id="test",
        category=TaskCategory.GENERAL_QA,
        task_name=fixture_id,
        provider="mock",
        model="mock",
        status=TaskStatus.SUCCESS,
        total_tokens=100,
        latency_ms=500.0,
        estimated_cost_usd=0.01,
        accuracy_score=overall,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )

    return ConversationResult(
        fixture=fixture,
        turn_results=turn_results,
        conversation_judge=conv_judge,
        task_result=tr,
        overall_score=overall,
    )


def test_summary_avg_turn_score():
    from datetime import UTC, datetime

    results = [_make_conversation_result(turn_scores=[0.8, 0.9], conv_score=0.85)]
    summary = MultiturnRunSummary(
        run_id="test", provider="mock", model="mock",
        judge_provider="mock", judge_model="mock",
        started_at=datetime.now(UTC), completed_at=datetime.now(UTC),
        conversation_results=results,
    )
    assert summary.avg_turn_score == 0.85


def test_summary_to_dict():
    from datetime import UTC, datetime

    results = [_make_conversation_result(turn_scores=[0.7], conv_score=0.8)]
    summary = MultiturnRunSummary(
        run_id="test", provider="mock", model="mock",
        judge_provider="mock", judge_model="mock",
        started_at=datetime.now(UTC), completed_at=datetime.now(UTC),
        conversation_results=results,
    )
    d = summary.to_dict()
    assert "avg_turn_score" in d
    assert "avg_conversation_score" in d
    assert "avg_retention" in d
    assert "avg_consistency" in d
    assert len(d["conversations"]) == 1


def test_summary_empty():
    from datetime import UTC, datetime

    summary = MultiturnRunSummary(
        run_id="test", provider="mock", model="mock",
        judge_provider="mock", judge_model="mock",
        started_at=datetime.now(UTC), completed_at=datetime.now(UTC),
    )
    assert summary.avg_turn_score is None
    assert summary.avg_conversation_score is None
    assert summary.total_turns == 0


# ── CLI tests ────────────────────────────────────────────────────────────────


def test_cli_multiturn_help():
    from click.testing import CliRunner
    from atomics.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["multiturn", "--help"])
    assert result.exit_code == 0
    assert "context retention" in result.output.lower() or "Multi-turn" in result.output
    assert "--judge-provider" in result.output
