"""RAG pipeline evaluation — grounding, faithfulness, and abstention scoring."""

from __future__ import annotations

from dataclasses import dataclass, field

from atomics.models import TaskComplexity


@dataclass(frozen=True)
class RAGChunk:
    """A single retrieved context chunk."""

    content: str
    label: str  # "relevant" | "distractor"
    source: str  # e.g. "CVE-2026-1234.md"


@dataclass(frozen=True)
class RAGFixture:
    """A RAG evaluation fixture with question, context chunks, and expectations."""

    id: str
    complexity: TaskComplexity
    question: str
    context_chunks: list[RAGChunk]
    gold_criteria: list[str] = field(default_factory=list)
    context_contains_answer: bool = True
    max_output_tokens: int = 512


__all__ = ["RAGChunk", "RAGFixture"]
