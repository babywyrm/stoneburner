"""Shared statistics helpers.

Single home for the percentile calculation that was previously copy-pasted across
the load-testing and storage modules. Kept dependency-free (no numpy) so it works
in the base install.
"""

from __future__ import annotations

from collections.abc import Iterable


def percentile(values: Iterable[float], pct: float) -> float:
    """Linear-interpolated percentile of `values` (any order).

    `pct` is 0–100. Returns 0.0 for an empty input. The input is sorted
    internally, so callers may pass values in any order.
    """
    s = sorted(values)
    if not s:
        return 0.0
    k = (len(s) - 1) * (pct / 100)
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])
