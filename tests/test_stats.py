"""Tests for the shared statistics helpers."""
from __future__ import annotations

import pytest

from atomics.stats import percentile


def test_percentile_empty_is_zero():
    assert percentile([], 50) == 0.0
    assert percentile([], 95) == 0.0


def test_percentile_single_value():
    assert percentile([42.0], 0) == 42.0
    assert percentile([42.0], 50) == 42.0
    assert percentile([42.0], 100) == 42.0


def test_percentile_known_values():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(data, 0) == 1.0
    assert percentile(data, 100) == 5.0
    assert percentile(data, 50) == 3.0  # median


def test_percentile_linear_interpolation():
    # 95th of 1..10 → index 0.95*9 = 8.55 → 9 + 0.55*(10-9) = 9.55
    data = list(map(float, range(1, 11)))
    assert percentile(data, 95) == pytest.approx(9.55)


def test_percentile_sorts_internally():
    """Unsorted input yields the same result as sorted input."""
    unsorted = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert percentile(unsorted, 50) == percentile(sorted(unsorted), 50)
    assert percentile(unsorted, 95) == percentile(sorted(unsorted), 95)


def test_percentile_matches_legacy_algorithm():
    """Guard: the shared impl matches the original inline formula."""
    def _legacy(values, pct):
        s = sorted(values)
        if not s:
            return 0.0
        k = (len(s) - 1) * (pct / 100)
        f = int(k)
        c = f + 1
        if c >= len(s):
            return s[f]
        return s[f] + (k - f) * (s[c] - s[f])

    import random
    rng = random.Random(1234)
    for _ in range(50):
        data = [rng.uniform(0, 1000) for _ in range(rng.randint(1, 40))]
        for pct in (0, 25, 50, 90, 95, 99, 100):
            assert percentile(data, pct) == _legacy(data, pct)
