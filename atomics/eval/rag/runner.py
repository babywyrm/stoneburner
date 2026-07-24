"""RAG eval runner — executes RAG fixtures and scores with the RAG judge."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from atomics.eval.rag import RAGFixture
from atomics.eval.rag.fixtures import ALL_RAG_FIXTURES
from atomics.eval.rag.judge import RAGJudgeResult, compute_hallucination, score_rag_response
from atomics.eval.rag.retrieval import RAGIndex
from atomics.models import TaskCategory, TaskResult, TaskStatus
from atomics.providers.base import BaseProvider
from atomics.validation import sanitize_error

logger = logging.getLogger("atomics.eval.rag.runner")


@dataclass
class RAGFixtureResult:
    fixture: RAGFixture
    task_result: TaskResult
    judge: RAGJudgeResult | None


@dataclass
class RAGRunSummary:
    run_id: str
    provider: str
    model: str
    judge_provider: str
    judge_model: str
    started_at: datetime
    completed_at: datetime
    fixture_results: list[RAGFixtureResult] = field(default_factory=list)
    avg_retrieved_chunks: float | None = None
    unique_sources_retrieved: int | None = None
    index_info: dict[str, Any] | None = None

    @property
    def grounding_score(self) -> float | None:
        scored = [
            r.judge.grounding
            for r in self.fixture_results
            if r.judge and not r.judge.parse_failed
        ]
        return round(sum(scored) / len(scored) / 4.0, 3) if scored else None

    @property
    def faithfulness_score(self) -> float | None:
        scored = [
            r.judge.faithfulness
            for r in self.fixture_results
            if r.judge and not r.judge.parse_failed
        ]
        return round(sum(scored) / len(scored) / 3.0, 3) if scored else None

    @property
    def abstention_accuracy(self) -> float | None:
        """Fraction of fixtures where the model made the correct abstain/answer decision."""
        scored = [r for r in self.fixture_results if r.judge and not r.judge.parse_failed]
        if not scored:
            return None
        correct = 0
        for r in scored:
            assert r.judge is not None
            if r.fixture.context_contains_answer:
                if r.judge.abstention >= 2:
                    correct += 1
            else:
                if r.judge.abstention >= 2:
                    correct += 1
        return round(correct / len(scored), 3)

    @property
    def hallucination_rate(self) -> float | None:
        results = [r for r in self.fixture_results if r.task_result.status == TaskStatus.SUCCESS]
        if not results:
            return None
        hallucinated = sum(
            1 for r in results if compute_hallucination(r.task_result.response, r.fixture)
        )
        return round(hallucinated / len(results), 3)

    @property
    def overall_rag_score(self) -> float | None:
        scored = [
            r.judge.score
            for r in self.fixture_results
            if r.judge and not r.judge.parse_failed
        ]
        return round(sum(scored) / len(scored), 3) if scored else None

    @property
    def total_cost_usd(self) -> float:
        return sum(r.task_result.estimated_cost_usd for r in self.fixture_results)

    @property
    def avg_latency_ms(self) -> float:
        lats = [r.task_result.latency_ms for r in self.fixture_results if r.task_result.latency_ms]
        return round(sum(lats) / len(lats), 1) if lats else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(r.task_result.total_tokens for r in self.fixture_results)

    @property
    def parse_failure_rate(self) -> float:
        judged = [r for r in self.fixture_results if r.judge is not None]
        if not judged:
            return 0.0
        failed = sum(1 for r in judged if r.judge is not None and r.judge.parse_failed)
        return round(failed / len(judged), 3)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "model": self.model,
            "judge_provider": self.judge_provider,
            "judge_model": self.judge_model,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "overall_rag_score": self.overall_rag_score,
            "grounding_score": self.grounding_score,
            "faithfulness_score": self.faithfulness_score,
            "abstention_accuracy": self.abstention_accuracy,
            "hallucination_rate": self.hallucination_rate,
            "parse_failure_rate": self.parse_failure_rate,
            "total_fixtures": len(self.fixture_results),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_tokens": self.total_tokens,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_retrieved_chunks": self.avg_retrieved_chunks,
            "unique_sources_retrieved": self.unique_sources_retrieved,
            "index_info": self.index_info,
            "fixtures": [
                {
                    "id": r.fixture.id,
                    "complexity": r.fixture.complexity.value,
                    "context_contains_answer": r.fixture.context_contains_answer,
                    "status": r.task_result.status.value,
                    "grounding": r.judge.grounding if r.judge else None,
                    "faithfulness": r.judge.faithfulness if r.judge else None,
                    "abstention": r.judge.abstention if r.judge else None,
                    "score": r.judge.score if r.judge else None,
                    "rationale": r.judge.rationale if r.judge else None,
                    "latency_ms": r.task_result.latency_ms,
                    "tokens": r.task_result.total_tokens,
                    "cost": round(r.task_result.estimated_cost_usd, 6),
                }
                for r in self.fixture_results
            ],
        }


def _build_rag_prompt(fixture: RAGFixture) -> str:
    """Build the prompt that includes the context chunks."""
    chunks_text = "\n\n".join(
        f"[Document {i}: {chunk.source}]\n{chunk.content}"
        for i, chunk in enumerate(fixture.context_chunks, 1)
    )
    return (
        f"You are given the following context documents to answer a question. "
        f"Base your answer ONLY on the provided context. If the context does not "
        f"contain enough information to answer the question, say so clearly.\n\n"
        f"CONTEXT:\n{chunks_text}\n\n"
        f"QUESTION: {fixture.question}"
    )


async def run_rag(
    provider: BaseProvider,
    *,
    judge_provider: BaseProvider | None = None,
    model: str | None = None,
    judge_model: str | None = None,
    run_id: str | None = None,
    on_fixture_done: Callable[[RAGFixtureResult], None] | None = None,
    thinking: bool | None = None,
    thinking_budget: int | None = None,
    fixtures: list[RAGFixture] | None = None,
    index: RAGIndex | None = None,
    top_k: int = 5,
) -> RAGRunSummary:
    """Run the RAG evaluation suite."""
    effective_judge = judge_provider or provider
    effective_run_id = run_id or uuid.uuid4().hex[:12]
    selected = fixtures if fixtures is not None else ALL_RAG_FIXTURES
    started = datetime.now(UTC)

    results: list[RAGFixtureResult] = []
    retrieved_chunk_counts: list[int] = []
    retrieved_sources: set[str] = set()

    for fixture in selected:
        if index is not None:
            retrieved_chunks = index.search(fixture.question, top_k=top_k)
            retrieved_chunk_counts.append(len(retrieved_chunks))
            retrieved_sources.update(c.source for c in retrieved_chunks)
            effective_fixture = RAGFixture(
                id=fixture.id,
                complexity=fixture.complexity,
                question=fixture.question,
                context_chunks=retrieved_chunks,
                gold_criteria=fixture.gold_criteria,
                context_contains_answer=fixture.context_contains_answer,
                max_output_tokens=fixture.max_output_tokens,
            )
        else:
            effective_fixture = fixture
        prompt = _build_rag_prompt(effective_fixture)
        task_id = uuid.uuid4().hex[:12]
        task_started = datetime.now(UTC)

        gen_kwargs: dict = {
            "model": model,
            "max_tokens": fixture.max_output_tokens,
        }
        if thinking is not None:
            gen_kwargs["thinking"] = thinking
        if thinking_budget is not None:
            gen_kwargs["thinking_budget"] = thinking_budget

        try:
            resp = await provider.generate(prompt, **gen_kwargs)
            tr = TaskResult(
                task_id=task_id,
                run_id=effective_run_id,
                category=TaskCategory.GENERAL_QA,
                task_name=fixture.id,
                provider=provider.name,
                model=resp.model or model or "",
                status=TaskStatus.SUCCESS,
                prompt=prompt,
                response=resp.text,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                total_tokens=resp.total_tokens,
                latency_ms=resp.latency_ms,
                estimated_cost_usd=resp.estimated_cost_usd,
                started_at=task_started,
                completed_at=datetime.now(UTC),
            )
        except Exception as exc:
            tr = TaskResult(
                task_id=task_id,
                run_id=effective_run_id,
                category=TaskCategory.GENERAL_QA,
                task_name=fixture.id,
                provider=provider.name,
                model=model or "",
                status=TaskStatus.FAILED,
                prompt=prompt,
                error_class=type(exc).__name__,
                error_message=sanitize_error(exc),
                started_at=task_started,
                completed_at=datetime.now(UTC),
            )
            fr = RAGFixtureResult(fixture=fixture, task_result=tr, judge=None)
            results.append(fr)
            if on_fixture_done:
                on_fixture_done(fr)
            continue

        judge_result = await score_rag_response(
            response=resp.text,
            fixture=effective_fixture,
            judge=effective_judge,
            judge_model=judge_model,
        )

        tr.accuracy_score = judge_result.score
        tr.quality_rationale = judge_result.rationale

        fr = RAGFixtureResult(fixture=fixture, task_result=tr, judge=judge_result)
        results.append(fr)
        if on_fixture_done:
            on_fixture_done(fr)

    completed = datetime.now(UTC)

    avg_retrieved: float | None = None
    unique_sources: int | None = None
    index_meta: dict[str, Any] | None = None
    if index is not None:
        avg_retrieved = (
            round(sum(retrieved_chunk_counts) / len(retrieved_chunk_counts), 3)
            if retrieved_chunk_counts
            else 0.0
        )
        unique_sources = len(retrieved_sources)
        index_meta = index.info()

    judge_prov_name = effective_judge.name if effective_judge else "none"
    return RAGRunSummary(
        run_id=effective_run_id,
        provider=provider.name,
        model=model or getattr(provider, "default_model", None) or "",
        judge_provider=judge_prov_name,
        judge_model=judge_model or "",
        started_at=started,
        completed_at=completed,
        fixture_results=results,
        avg_retrieved_chunks=avg_retrieved,
        unique_sources_retrieved=unique_sources,
        index_info=index_meta,
    )
