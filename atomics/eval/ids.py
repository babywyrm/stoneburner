"""Subset a fixture catalog by id. Unknown ids raise so a typo cannot run all."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def select_by_ids(catalog: Sequence[T], ids: Sequence[str] | None) -> list[T]:
    """Keep request order. Empty ``ids`` after strip is ``no fixture IDs``."""
    if ids is None:
        return list(catalog)
    wanted = [item.strip() for item in ids if item.strip()]
    if not wanted:
        raise ValueError("no fixture IDs")
    by_id = {getattr(fixture, "id"): fixture for fixture in catalog}
    missing = [item for item in wanted if item not in by_id]
    if missing:
        raise ValueError(f"unknown fixture IDs: {', '.join(missing)}")
    return [by_id[item] for item in wanted]
