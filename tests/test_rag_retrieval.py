import json
import math
from pathlib import Path

import pytest

try:
    import sqlite_vec
except ImportError:
    sqlite_vec = None  # type: ignore[assignment]

from atomics.eval.rag.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from atomics.eval.rag.retrieval import (
    Document,
    MockEmbedder,
    RAGIndex,
    chunk_text,
    load_documents,
)


def test_chunk_text_splits_and_overlaps():
    text = "abcdefghijklmnopqrstuvwxyz" * 10  # 260 chars
    chunk_size = 100
    overlap = 20
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert len(chunk) == chunk_size
    # Last chunk may be shorter
    assert len(chunks[-1]) <= chunk_size
    # Second chunk should start before the end of the first (true character overlap)
    assert chunks[1].startswith(chunks[0][-overlap:])
    assert chunks[0][chunk_size - overlap :] == chunks[1][:overlap]


def test_chunk_text_empty_returns_empty():
    assert chunk_text("") == []


def test_chunk_text_expected_sizes():
    text = "0123456789" * 5  # 50 chars
    chunks = chunk_text(text, chunk_size=10, overlap=3)
    assert chunks == [
        "0123456789",
        "7890123456",
        "4567890123",
        "1234567890",
        "8901234567",
        "5678901234",
        "23456789",
    ]
    for chunk in chunks[:-1]:
        assert len(chunk) == 10
    assert len(chunks[-1]) == 8


def test_chunk_text_invalid_overlap_raises():
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("hello", chunk_size=10, overlap=10)
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("hello", chunk_size=10, overlap=-1)


def test_mock_embedder_dim_and_embedding():
    embedder = MockEmbedder(dim=8)
    assert embedder.dim == 8
    vectors = embedder.embed(["hello", "world"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 8
    assert vectors[0] == vectors[1]  # mock returns identical deterministic vectors


def test_load_documents_from_directory(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello world")
    (tmp_path / "b.md").write_text("# Title\n\ncontent")
    docs = load_documents(tmp_path)
    assert len(docs) == 2
    sources = {d.source for d in docs}
    assert sources == {"a.txt", "b.md"}
    by_source = {d.source: d for d in docs}
    assert by_source["a.txt"].metadata is not None
    assert by_source["a.txt"].metadata["full_path"] == str((tmp_path / "a.txt").resolve())
    assert by_source["b.md"].metadata is not None
    assert by_source["b.md"].metadata["full_path"] == str((tmp_path / "b.md").resolve())


def test_load_documents_single_file(tmp_path: Path):
    path = tmp_path / "solo.txt"
    path.write_text("single file content")
    docs = load_documents(path)
    assert len(docs) == 1
    assert docs[0].content == "single file content"
    assert docs[0].source == "solo.txt"
    assert docs[0].metadata is not None
    assert docs[0].metadata["full_path"] == str(path.resolve())


def test_load_documents_returns_basenames(tmp_path: Path):
    path = tmp_path / "CVE-2026-3891.md"
    path.write_text("# CVE advisory\n\nDetails here.")
    docs = load_documents(path)
    assert len(docs) == 1
    assert docs[0].source == "CVE-2026-3891.md"
    assert docs[0].metadata is not None
    assert docs[0].metadata["full_path"] == str(path.resolve())


def test_load_documents_json_text_field(tmp_path: Path):
    path = tmp_path / "doc.json"
    path.write_text(json.dumps({"text": "json text content", "id": 1}))
    docs = load_documents(path)
    assert len(docs) == 1
    assert docs[0].content == "json text content"
    assert isinstance(docs[0].content, str)


def test_load_documents_json_non_string_text_coerced(tmp_path: Path):
    path = tmp_path / "doc.json"
    path.write_text(json.dumps({"text": ["a", "b"]}))
    docs = load_documents(path)
    assert len(docs) == 1
    assert isinstance(docs[0].content, str)
    assert docs[0].content == json.dumps(["a", "b"])


def test_load_documents_missing_path_raises(tmp_path: Path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="Path is not a file or directory"):
        load_documents(missing)


class _DistinctEmbedder:
    """Test embedder with content-dependent vectors so ranking can be asserted."""

    dim = 3
    model_name = "distinct-mock"

    _vectors: dict[str, list[float]] = {
        "the quick brown fox": [1.0, 0.0, 0.0],
        "lazy dog jumps": [0.0, 1.0, 0.0],
        "fox": [1.0, 0.0, 0.0],
    }

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors.get(t, [0.0, 0.0, 1.0]) for t in texts]


@pytest.mark.skipif(sqlite_vec is None, reason="sqlite-vec not installed")
def test_rag_index_build_and_search(tmp_path: Path):
    db_path = tmp_path / "index.vec"
    embedder = MockEmbedder(dim=8)
    index = RAGIndex(db_path, embedder=embedder)
    docs = [
        Document(content="the quick brown fox", source="a.txt"),
        Document(content="lazy dog jumps", source="b.txt"),
    ]
    count = index.build(docs, chunk_size=20, overlap=0)
    assert count == 2

    results = index.search("fox", top_k=2)
    assert len(results) == 2
    assert all(r.source in ("a.txt", "b.txt") for r in results)
    assert all(r.label == "retrieved" for r in results)
    assert index.info()["chunk_count"] == "2"
    assert index.info()["embedding_model"] == "mock"


@pytest.mark.skipif(sqlite_vec is None, reason="sqlite-vec not installed")
def test_rag_index_round_trips_basename_sources(tmp_path: Path):
    (tmp_path / "CVE-2026-3891.md").write_text("advisory body about CVE-2026-3891")
    docs = load_documents(tmp_path)
    assert docs[0].source == "CVE-2026-3891.md"

    db_path = tmp_path / "index.vec"
    index = RAGIndex(db_path, embedder=MockEmbedder(dim=8))
    index.build(docs, chunk_size=64, overlap=0)
    results = index.search("CVE-2026-3891", top_k=1)
    assert len(results) == 1
    assert results[0].source == "CVE-2026-3891.md"
    info = index.info()
    assert info["embedding_model"] == "mock"
    source_paths = json.loads(info["source_paths"])
    assert source_paths["CVE-2026-3891.md"] == str(
        (tmp_path / "CVE-2026-3891.md").resolve()
    )


@pytest.mark.skipif(sqlite_vec is None, reason="sqlite-vec not installed")
def test_rag_index_empty_build_clears_stale_index(tmp_path: Path):
    db_path = tmp_path / "index.vec"
    index = RAGIndex(db_path, embedder=MockEmbedder(dim=8))
    index.build(
        [Document(content="stale content stays until cleared", source="old.txt")],
        chunk_size=64,
        overlap=0,
    )
    assert index.info()["chunk_count"] == "1"

    count = index.build([], chunk_size=32, overlap=0)
    assert count == 0
    assert index.search("stale", top_k=5) == []
    info = index.info()
    assert info["chunk_count"] == "0"
    assert info["chunk_size"] == "32"
    assert info["overlap"] == "0"
    assert info["embedding_model"] == "mock"


@pytest.mark.skipif(sqlite_vec is None, reason="sqlite-vec not installed")
def test_rag_index_search_ranks_by_embedding_distance(tmp_path: Path):
    db_path = tmp_path / "index.vec"
    index = RAGIndex(db_path, embedder=_DistinctEmbedder())
    index.build(
        [
            Document(content="the quick brown fox", source="fox.txt"),
            Document(content="lazy dog jumps", source="dog.txt"),
        ],
        chunk_size=64,
        overlap=0,
    )
    results = index.search("fox", top_k=2)
    assert len(results) == 2
    assert results[0].source == "fox.txt"
    assert results[1].source == "dog.txt"


def test_recall_at_k():
    assert recall_at_k({"a", "b"}, ["a", "c", "d"], 2) == 0.5
    assert recall_at_k({"a", "b"}, ["a", "b", "c"], 3) == 1.0


def test_precision_at_k():
    assert precision_at_k({"a", "b"}, ["a", "c", "d"], 2) == 0.5
    assert precision_at_k({"a", "b"}, ["a", "b"], 2) == 1.0


def test_mrr():
    assert mean_reciprocal_rank([{"a"}], [["c", "a"]]) == 0.5
    assert mean_reciprocal_rank([{"a"}], [["a", "b"]]) == 1.0


def test_ndcg_at_k():
    relevance = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert ndcg_at_k(relevance, ["a", "b"], 2) == 1.0


def test_metrics_k_zero_returns_zero():
    assert recall_at_k({"a"}, ["a", "b"], 0) == 0.0
    assert precision_at_k({"a"}, ["a", "b"], 0) == 0.0
    assert ndcg_at_k({"a": 1.0}, ["a", "b"], 0) == 0.0


def test_metrics_negative_k_raises():
    with pytest.raises(ValueError, match="k must be >= 0"):
        recall_at_k({"a"}, ["a"], -1)
    with pytest.raises(ValueError, match="k must be >= 0"):
        precision_at_k({"a"}, ["a"], -1)
    with pytest.raises(ValueError, match="k must be >= 0"):
        ndcg_at_k({"a": 1.0}, ["a"], -1)


def test_metrics_empty_relevant_and_retrieved():
    assert recall_at_k(set(), [], 3) == 1.0
    assert precision_at_k(set(), [], 3) == 0.0
    assert recall_at_k(set(), ["a"], 3) == 0.0
    assert precision_at_k(set(), ["a"], 3) == 0.0


def test_metrics_k_larger_than_retrieved():
    assert recall_at_k({"a", "b"}, ["a"], 5) == 0.5
    assert precision_at_k({"a", "b"}, ["a"], 5) == 1.0
    # Ideal DCG includes both graded items even though only one was retrieved
    actual_dcg = 2**3 - 1  # log2(2) == 1
    ideal_dcg = (2**3 - 1) + (2**2 - 1) / math.log2(3)
    assert ndcg_at_k({"a": 3.0, "b": 2.0}, ["a"], 5) == pytest.approx(actual_dcg / ideal_dcg)


def test_mrr_no_hit_returns_zero():
    assert mean_reciprocal_rank([{"a"}], [["b", "c"]]) == 0.0


def test_mrr_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        mean_reciprocal_rank([{"a"}], [["a"], ["b"]])
    with pytest.raises(ValueError, match="same length"):
        mean_reciprocal_rank([], [])


def test_ndcg_non_ideal_ranking():
    relevance = {"a": 3.0, "b": 2.0, "c": 1.0}
    # Perfect: [a, b] -> 1.0; reversed relevance order [c, b] is worse
    perfect = ndcg_at_k(relevance, ["a", "b"], 2)
    imperfect = ndcg_at_k(relevance, ["c", "b"], 2)
    assert perfect == 1.0
    assert 0.0 < imperfect < 1.0


def test_ndcg_empty_relevance_scores():
    assert ndcg_at_k({}, ["a", "b"], 2) == 0.0
