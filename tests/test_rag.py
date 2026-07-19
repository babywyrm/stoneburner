"""Tests for the RAG pipeline evaluation suite."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from atomics.eval.rag import RAGChunk, RAGFixture
from atomics.eval.rag.fixtures import ALL_RAG_FIXTURES
from atomics.eval.rag.judge import (
    RAGJudgeResult,
    _parse_rag_rubric,
    compute_hallucination,
)
from atomics.eval.rag.runner import RAGFixtureResult, RAGRunSummary, _build_rag_prompt, run_rag
from atomics.models import TaskCategory, TaskComplexity, TaskResult, TaskStatus
from atomics.providers.base import ProviderResponse

try:
    import sqlite_vec
except ImportError:
    sqlite_vec = None  # type: ignore[assignment]

# ── Fixture model tests ──────────────────────────────────────────────────────


def test_rag_chunk_creation():
    chunk = RAGChunk(content="test content", label="relevant", source="test.md")
    assert chunk.label == "relevant"
    assert chunk.source == "test.md"


def test_rag_fixture_defaults():
    f = RAGFixture(
        id="test-1",
        complexity=TaskComplexity.LIGHT,
        question="What?",
        context_chunks=[],
    )
    assert f.context_contains_answer is True
    assert f.max_output_tokens == 512


def test_rag_fixture_abstention():
    f = RAGFixture(
        id="test-2",
        complexity=TaskComplexity.LIGHT,
        question="What?",
        context_chunks=[],
        context_contains_answer=False,
    )
    assert f.context_contains_answer is False


# ── Fixtures collection tests ────────────────────────────────────────────────


def test_all_fixtures_loaded():
    assert len(ALL_RAG_FIXTURES) == 20


def test_fixture_ids_are_unique():
    ids = [f.id for f in ALL_RAG_FIXTURES]
    assert len(ids) == len(set(ids))


def test_fixture_ids_follow_convention():
    for f in ALL_RAG_FIXTURES:
        assert f.id.startswith("rag-"), f"Fixture {f.id} doesn't follow rag-NN pattern"


def test_fixtures_have_context_chunks():
    for f in ALL_RAG_FIXTURES:
        assert len(f.context_chunks) >= 1, f"Fixture {f.id} has no context chunks"


def test_fixtures_have_gold_criteria():
    for f in ALL_RAG_FIXTURES:
        assert len(f.gold_criteria) >= 2, f"Fixture {f.id} has fewer than 2 gold criteria"


def test_abstention_fixtures_exist():
    abstention = [f for f in ALL_RAG_FIXTURES if not f.context_contains_answer]
    assert len(abstention) >= 3, "Expected at least 3 abstention fixtures"


def test_security_and_general_mix():
    security_ids = [f.id for f in ALL_RAG_FIXTURES if int(f.id.split("-")[1]) <= 10]
    general_ids = [f.id for f in ALL_RAG_FIXTURES if int(f.id.split("-")[1]) > 10]
    assert len(security_ids) == 10
    assert len(general_ids) == 10


def test_complexity_spread():
    complexities = {f.complexity for f in ALL_RAG_FIXTURES}
    assert TaskComplexity.LIGHT in complexities
    assert TaskComplexity.MODERATE in complexities
    assert TaskComplexity.HEAVY in complexities


def test_chunks_have_labels():
    for f in ALL_RAG_FIXTURES:
        for chunk in f.context_chunks:
            assert chunk.label in ("relevant", "distractor"), (
                f"Fixture {f.id} chunk from {chunk.source} has invalid label: {chunk.label}"
            )


# ── Judge parse tests ────────────────────────────────────────────────────────


def test_parse_strict_format():
    raw = "GROUNDING: 3\nFAITHFULNESS: 2\nABSTENTION: 3\nRATIONALE: Well grounded response."
    result = _parse_rag_rubric(raw)
    assert result is not None
    grounding, faithfulness, abstention, rationale = result
    assert grounding == 3
    assert faithfulness == 2
    assert abstention == 3
    assert "grounded" in rationale.lower()


def test_parse_lenient_format():
    raw = "**Grounding** - 4\n**Faithfulness** - 3\nAbstention: 2\nRationale: Good job."
    result = _parse_rag_rubric(raw)
    assert result is not None
    assert result[0] == 4
    assert result[1] == 3
    assert result[2] == 2


def test_parse_clamped_values():
    raw = "GROUNDING: 9\nFAITHFULNESS: 7\nABSTENTION: 5\nRATIONALE: Over-scored."
    result = _parse_rag_rubric(raw)
    assert result is not None
    assert result[0] == 4  # clamped
    assert result[1] == 3  # clamped
    assert result[2] == 3  # clamped


def test_parse_returns_none_on_garbage():
    assert _parse_rag_rubric("This is just random text") is None


def test_parse_crlf():
    raw = "GROUNDING: 2\r\nFAITHFULNESS: 1\r\nABSTENTION: 3\r\nRATIONALE: OK."
    result = _parse_rag_rubric(raw)
    assert result is not None
    assert result[0] == 2


def test_parse_missing_rationale():
    raw = "Grounding: 3\nFaithfulness: 2\nAbstention: 1"
    result = _parse_rag_rubric(raw)
    assert result is not None
    assert result[3] == "(no rationale provided)"


# ── Hallucination heuristic tests ────────────────────────────────────────────


def _abstention_fixture() -> RAGFixture:
    return RAGFixture(
        id="test-abs",
        complexity=TaskComplexity.LIGHT,
        question="What?",
        context_chunks=[RAGChunk("irrelevant", "distractor", "x.md")],
        context_contains_answer=False,
    )


def _answer_fixture() -> RAGFixture:
    return RAGFixture(
        id="test-ans",
        complexity=TaskComplexity.LIGHT,
        question="What?",
        context_chunks=[RAGChunk("the answer is here", "relevant", "x.md")],
        context_contains_answer=True,
    )


def test_hallucination_correct_abstention():
    assert (
        compute_hallucination(
            "I cannot answer this from the given context.",
            _abstention_fixture(),
        )
        is False
    )


def test_hallucination_false_answer_on_abstention():
    assert (
        compute_hallucination(
            "Based on the analysis, the vulnerability affects "
            "version 3.2 and requires immediate patching.",
            _abstention_fixture(),
        )
        is True
    )


def test_hallucination_short_response_not_flagged():
    assert compute_hallucination("No.", _abstention_fixture()) is False


def test_hallucination_answer_fixture_not_flagged():
    assert compute_hallucination("Anything here is fine", _answer_fixture()) is False


# ── Prompt builder tests ─────────────────────────────────────────────────────


def test_build_rag_prompt_includes_question():
    fixture = ALL_RAG_FIXTURES[0]
    prompt = _build_rag_prompt(fixture)
    assert fixture.question in prompt


def test_build_rag_prompt_includes_context():
    fixture = ALL_RAG_FIXTURES[0]
    prompt = _build_rag_prompt(fixture)
    for chunk in fixture.context_chunks:
        assert chunk.source in prompt
        assert chunk.content[:50] in prompt


def test_build_rag_prompt_includes_instructions():
    fixture = ALL_RAG_FIXTURES[0]
    prompt = _build_rag_prompt(fixture)
    assert "ONLY on the provided context" in prompt


def test_build_rag_prompt_with_retrieved_context():
    fixture = RAGFixture(
        id="test-retrieval",
        complexity=TaskComplexity.LIGHT,
        question="what is the answer",
        context_chunks=[RAGChunk(content="the answer is 42", label="retrieved", source="doc.txt")],
    )
    prompt = _build_rag_prompt(fixture)
    assert "the answer is 42" in prompt
    assert "[Document 1: doc.txt]" in prompt


# ── Summary tests ────────────────────────────────────────────────────────────


def _make_fixture_result(
    fixture_id: str = "rag-01",
    grounding: int = 3,
    faithfulness: int = 2,
    abstention: int = 3,
    status: TaskStatus = TaskStatus.SUCCESS,
    context_contains_answer: bool = True,
) -> RAGFixtureResult:
    from datetime import UTC, datetime

    fixture = RAGFixture(
        id=fixture_id,
        complexity=TaskComplexity.LIGHT,
        question="test",
        context_chunks=[],
        context_contains_answer=context_contains_answer,
    )
    score = (grounding + faithfulness + abstention) / 10.0
    tr = TaskResult(
        run_id="test-run",
        category=TaskCategory.GENERAL_QA,
        task_name=fixture_id,
        provider="mock",
        model="mock",
        status=status,
        response="test response",
        total_tokens=100,
        latency_ms=500.0,
        estimated_cost_usd=0.01,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    judge = RAGJudgeResult(
        grounding=grounding,
        faithfulness=faithfulness,
        abstention=abstention,
        score=score,
        rationale="test rationale",
    )
    return RAGFixtureResult(fixture=fixture, task_result=tr, judge=judge)


def test_summary_overall_rag_score():
    from datetime import UTC, datetime

    results = [_make_fixture_result(grounding=4, faithfulness=3, abstention=3)]
    summary = RAGRunSummary(
        run_id="test",
        provider="mock",
        model="mock",
        judge_provider="mock",
        judge_model="mock",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        fixture_results=results,
    )
    assert summary.overall_rag_score == 1.0


def test_summary_grounding_score():
    from datetime import UTC, datetime

    results = [
        _make_fixture_result(grounding=4),
        _make_fixture_result(fixture_id="rag-02", grounding=2),
    ]
    summary = RAGRunSummary(
        run_id="test",
        provider="mock",
        model="mock",
        judge_provider="mock",
        judge_model="mock",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        fixture_results=results,
    )
    assert summary.grounding_score == 0.75  # (4+2)/2 / 4


def test_summary_to_dict():
    from datetime import UTC, datetime

    results = [_make_fixture_result()]
    summary = RAGRunSummary(
        run_id="test",
        provider="mock",
        model="mock",
        judge_provider="mock",
        judge_model="mock",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        fixture_results=results,
    )
    d = summary.to_dict()
    assert "overall_rag_score" in d
    assert "grounding_score" in d
    assert "faithfulness_score" in d
    assert "abstention_accuracy" in d
    assert "hallucination_rate" in d
    assert len(d["fixtures"]) == 1


def test_summary_empty():
    from datetime import UTC, datetime

    summary = RAGRunSummary(
        run_id="test",
        provider="mock",
        model="mock",
        judge_provider="mock",
        judge_model="mock",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    assert summary.overall_rag_score is None
    assert summary.grounding_score is None
    assert summary.hallucination_rate is None
    assert summary.parse_failure_rate == 0.0
    assert summary.avg_retrieved_chunks is None
    assert summary.unique_sources_retrieved is None
    assert summary.index_info is None


def test_summary_with_index_stats():
    results = [_make_fixture_result()]
    summary = RAGRunSummary(
        run_id="test",
        provider="mock",
        model="mock",
        judge_provider="mock",
        judge_model="mock",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        fixture_results=results,
        avg_retrieved_chunks=3.5,
        unique_sources_retrieved=2,
        index_info={"chunk_count": "10", "embedding_model": "mock"},
    )
    assert summary.avg_retrieved_chunks == 3.5
    assert summary.unique_sources_retrieved == 2
    assert summary.index_info == {"chunk_count": "10", "embedding_model": "mock"}
    d = summary.to_dict()
    assert d["avg_retrieved_chunks"] == 3.5
    assert d["unique_sources_retrieved"] == 2
    assert d["index_info"]["embedding_model"] == "mock"


# ── run_rag with index ───────────────────────────────────────────────────────


class _FakeIndex:
    def __init__(self, chunks: list[RAGChunk]) -> None:
        self._chunks = chunks
        self.search_calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[RAGChunk]:
        self.search_calls.append((query, top_k))
        return self._chunks[:top_k]

    def info(self) -> dict:
        return {"chunk_count": str(len(self._chunks)), "embedding_model": "mock"}


@pytest.mark.asyncio
async def test_run_rag_judge_receives_retrieved_chunks():
    retrieved = [
        RAGChunk(content="retrieved advisory text", label="retrieved", source="CVE-2026-3891.md"),
    ]
    index = _FakeIndex(retrieved)
    fixture = RAGFixture(
        id="rag-test",
        complexity=TaskComplexity.LIGHT,
        question="What is the CVE?",
        context_chunks=[RAGChunk(content="hand-crafted", label="relevant", source="hand.md")],
        gold_criteria=["mentions CVE"],
    )

    provider = AsyncMock()
    provider.name = "mock"
    provider.generate = AsyncMock(
        return_value=ProviderResponse(
            text="The CVE is described in the advisory.",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            model="mock-model",
            latency_ms=5.0,
            estimated_cost_usd=0.0,
        )
    )

    judge = AsyncMock()
    judge.name = "mock-judge"
    judge.generate = AsyncMock(
        return_value=SimpleNamespace(
            text="GROUNDING: 3\nFAITHFULNESS: 2\nABSTENTION: 3\nRATIONALE: Retrieved context used.",
            model="judge-model",
            input_tokens=5,
            output_tokens=5,
            total_tokens=10,
            latency_ms=1.0,
            estimated_cost_usd=0.0,
        )
    )

    summary = await run_rag(
        provider,
        judge_provider=judge,
        fixtures=[fixture],
        index=index,  # type: ignore[arg-type]
        top_k=3,
    )

    assert len(summary.fixture_results) == 1
    assert summary.fixture_results[0].fixture is fixture
    assert summary.fixture_results[0].judge is not None
    assert not summary.fixture_results[0].judge.parse_failed
    judge_prompt = judge.generate.call_args.args[0]
    assert "retrieved advisory text" in judge_prompt
    assert "CVE-2026-3891.md" in judge_prompt
    assert "hand-crafted" not in judge_prompt
    assert summary.avg_retrieved_chunks == 1.0
    assert summary.unique_sources_retrieved == 1
    assert summary.index_info is not None
    assert summary.index_info["embedding_model"] == "mock"


@pytest.mark.skipif(sqlite_vec is None, reason="sqlite-vec not installed")
@pytest.mark.asyncio
async def test_run_rag_with_real_index(tmp_path):
    from atomics.eval.rag.retrieval import Document, MockEmbedder, RAGIndex

    db_path = tmp_path / "index.vec"
    index = RAGIndex(db_path, embedder=MockEmbedder(dim=8))
    index.build(
        [
            Document(
                content="The vulnerability CVE-2026-3891 affects the auth service.",
                source="CVE-2026-3891.md",
                metadata={"full_path": "/tmp/CVE-2026-3891.md"},
            ),
        ],
        chunk_size=128,
        overlap=0,
    )

    fixture = RAGFixture(
        id="rag-test-index",
        complexity=TaskComplexity.LIGHT,
        question="What does CVE-2026-3891 affect?",
        context_chunks=[RAGChunk(content="placeholder", label="relevant", source="hand.md")],
        gold_criteria=["auth service"],
    )

    provider = AsyncMock()
    provider.name = "mock"
    provider.generate = AsyncMock(
        return_value=ProviderResponse(
            text="It affects the auth service.",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            model="mock-model",
            latency_ms=5.0,
            estimated_cost_usd=0.0,
        )
    )
    judge = AsyncMock()
    judge.name = "mock-judge"
    judge.generate = AsyncMock(
        return_value=SimpleNamespace(
            text="GROUNDING: 4\nFAITHFULNESS: 3\nABSTENTION: 3\nRATIONALE: Grounded.",
            model="judge-model",
            input_tokens=5,
            output_tokens=5,
            total_tokens=10,
            latency_ms=1.0,
            estimated_cost_usd=0.0,
        )
    )

    summary = await run_rag(
        provider,
        judge_provider=judge,
        fixtures=[fixture],
        index=index,
        top_k=2,
    )

    assert len(summary.fixture_results) == 1
    fr = summary.fixture_results[0]
    assert fr.judge is not None
    assert fr.judge.score > 0
    assert not fr.judge.parse_failed
    prompt = provider.generate.call_args.args[0]
    assert "CVE-2026-3891" in prompt
    assert "auth service" in prompt
    assert summary.avg_retrieved_chunks is not None
    assert summary.avg_retrieved_chunks >= 1
    assert summary.unique_sources_retrieved == 1
    assert summary.index_info is not None
    assert summary.index_info["embedding_model"] == "mock"


# ── CLI tests ────────────────────────────────────────────────────────────────


def test_cli_rag_help():
    from click.testing import CliRunner

    from atomics.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["rag", "--help"])
    assert result.exit_code == 0
    assert "grounding" in result.output.lower() or "RAG" in result.output
    assert "--judge-provider" in result.output
    assert "--fixtures" in result.output
