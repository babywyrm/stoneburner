"""95% confidence intervals for suite headlines.

A headline over 20 fixtures is an estimate, not a measurement. These put a
range on it. The unit is the fixture: with ``--runs``, a fixture's runs are
averaged first, so repeating a run never narrows the interval.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

_Z = 1.959964

# Two-sided 95% Student-t critical values for df 1-30.
# ponytail: above 30 this uses z, about 2% narrow at df 31 and closing; add
# rows (or a quantile function) if suites grow past 30 fixtures.
_T95 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)  # fmt: skip


@dataclass(frozen=True)
class Interval:
    low: float
    high: float
    n: int
    method: str  # "wilson" | "t"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def format(self) -> str:
        return f"95% CI {self.low * 100:.0f}–{self.high * 100:.0f}%, n={self.n}"


def _interval(low: float, high: float, n: int, method: str) -> Interval:
    return Interval(round(max(0.0, low), 4), round(min(1.0, high), 4), n, method)


def wilson(successes: int, n: int) -> Interval | None:
    """Wilson score interval for ``successes`` out of ``n``."""
    if n <= 0:
        return None
    p = successes / n
    z2 = _Z * _Z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = _Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return _interval(centre - half, centre + half, n, "wilson")


def mean_interval(
    values: Sequence[float], weights: Sequence[float] | None = None
) -> Interval | None:
    """Student-t interval for a (weighted) mean of 0-1 scores, clipped to [0, 1].

    Weighted: the standard error is the weighted form that reduces to s/sqrt(n)
    at equal weights, and degrees of freedom come from Kish's effective n.
    """
    n = len(values)
    w = [1.0] * n if weights is None else list(weights)
    if len(w) != n:
        raise ValueError(f"{len(w)} weights for {n} values")
    total = sum(w)
    if n < 2 or total <= 0:
        return None
    effective_n = total * total / sum(x * x for x in w)
    if effective_n < 2:
        return None
    mean = sum(x * v for x, v in zip(w, values, strict=True)) / total
    spread = sum(x * x * (v - mean) ** 2 for x, v in zip(w, values, strict=True))
    df = math.floor(effective_n) - 1
    # Zero spread gives t no width, and 5/5 is not certainty.
    # ponytail: a unanimous fractional score (all 0.8) stays zero-width;
    # bounding it needs a per-fixture variance model.
    if spread == 0 and mean in (0.0, 1.0) and (bound := wilson(round(mean) * (df + 1), df + 1)):
        return _interval(bound.low, bound.high, n, "wilson")
    se = math.sqrt(n / (n - 1) * spread) / total
    t = _T95[df - 1] if df <= len(_T95) else _Z
    return _interval(mean - t * se, mean + t * se, n, "t")


def rate_interval(per_fixture: Sequence[float]) -> Interval | None:
    """Interval for a rate: Wilson on 0/1 fixtures, t on per-fixture fractions."""
    if not per_fixture:
        return None
    if all(v in (0, 1) for v in per_fixture):
        return wilson(sum(1 for v in per_fixture if v == 1), len(per_fixture))
    return mean_interval(per_fixture)
