"""Retrieval-quality metrics for RAG evaluation."""

from __future__ import annotations

import math


def _validate_k(k: int) -> None:
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Fraction of relevant items found in the top-k retrieved items."""
    _validate_k(k)
    if k == 0:
        return 0.0
    if not relevant:
        return 1.0 if not retrieved[:k] else 0.0
    retrieved_k = set(retrieved[:k])
    return len(relevant & retrieved_k) / len(relevant)


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Fraction of top-k retrieved items that are relevant."""
    _validate_k(k)
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    return len(relevant & set(retrieved_k)) / len(retrieved_k)


def mean_reciprocal_rank(
    relevant_sets: list[set[str]],
    retrieved_lists: list[list[str]],
) -> float:
    """Mean reciprocal rank of the first relevant item in each retrieved list."""
    if not relevant_sets or not retrieved_lists or len(relevant_sets) != len(retrieved_lists):
        raise ValueError("relevant_sets and retrieved_lists must have the same length")
    rr_sum = 0.0
    for relevant, retrieved in zip(relevant_sets, retrieved_lists):
        for rank, item in enumerate(retrieved, start=1):
            if item in relevant:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(relevant_sets)


def ndcg_at_k(relevance_scores: dict[str, float], retrieved: list[str], k: int) -> float:
    """Normalized discounted cumulative gain at k."""
    _validate_k(k)
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0

    def dcg(items: list[str]) -> float:
        return sum(
            (2 ** relevance_scores.get(item, 0.0) - 1) / math.log2(i + 2)
            for i, item in enumerate(items)
        )

    ideal = sorted(relevance_scores.values(), reverse=True)[:k]
    ideal_dcg = sum((2 ** score - 1) / math.log2(i + 2) for i, score in enumerate(ideal))
    actual_dcg = dcg(retrieved_k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
