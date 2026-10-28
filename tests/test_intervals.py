"""95% confidence intervals for suite headlines."""

from __future__ import annotations

import pytest

from atomics.eval.intervals import (
    Interval,
    mean_interval,
    paired_interval,
    rate_interval,
    wilson,
)


def _bounds(interval: Interval | None) -> tuple[float, float]:
    assert interval is not None
    return interval.low, interval.high


@pytest.mark.parametrize(
    ("successes", "n", "low", "high"),
    [
        (6, 20, 0.1455, 0.5190),
        (0, 20, 0.0, 0.1611),
        (20, 20, 0.8389, 1.0),
        (1, 2, 0.0945, 0.9055),
    ],
)
def test_wilson_matches_reference_values(successes, n, low, high):
    assert _bounds(wilson(successes, n)) == pytest.approx((low, high), abs=1e-4)


def test_wilson_reports_its_method_and_n():
    interval = wilson(6, 20)
    assert interval is not None
    assert (interval.n, interval.method) == (20, "wilson")
    assert interval.to_dict() == {"low": 0.1455, "high": 0.519, "n": 20, "method": "wilson"}


def test_wilson_on_nothing_is_none():
    assert wilson(0, 0) is None


def test_mean_interval_uses_student_t_and_clips_to_unit_range():
    # mean 0.4, sd 0.2, se 0.1155, t(df=2) 4.303
    assert _bounds(mean_interval([0.2, 0.4, 0.6])) == pytest.approx((0.0, 0.8969), abs=1e-4)


def test_mean_interval_matches_t_for_df_9():
    values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.6, 0.7, 0.8, 0.9]
    # mean 0.7, sd 0.1491, se 0.04714, t(df=9) 2.262
    assert _bounds(mean_interval(values)) == pytest.approx((0.5934, 0.8066), abs=1e-4)


def test_mean_interval_falls_back_to_z_above_the_table():
    values = [0.0, 1.0] * 20  # n 40, mean 0.5, sd 0.5064, se 0.0801
    assert _bounds(mean_interval(values)) == pytest.approx((0.343, 0.657), abs=1e-3)


def test_equal_weights_match_unweighted():
    values = [0.2, 0.9, 0.4, 0.7]
    assert mean_interval(values, [3.0] * 4) == mean_interval(values)


def test_weighted_mean_uses_weighted_se_and_kish_n():
    # mean 0.4; se sqrt(4/3 * 0.48) / 6 = 0.1333; Kish n 36 / 12 = 3, t(df=2) 4.303
    interval = mean_interval([0.2, 0.6, 0.6, 0.6], [3.0, 1.0, 1.0, 1.0])
    assert _bounds(interval) == pytest.approx((0.0, 0.9737), abs=1e-4)
    assert interval is not None and (interval.n, interval.method) == (4, "t")


def test_weights_too_uneven_to_bound_give_none():
    # Kish n 121 / 101 = 1.2, under 2.
    assert mean_interval([0.0, 1.0], [10.0, 1.0]) is None


def test_mean_interval_needs_two_values_and_matching_weights():
    assert mean_interval([]) is None
    assert mean_interval([0.5]) is None
    with pytest.raises(ValueError, match="weights"):
        mean_interval([0.1, 0.2], [1.0])


def test_identical_scores_give_a_zero_width_interval():
    assert _bounds(mean_interval([0.8, 0.8, 0.8])) == (0.8, 0.8)


@pytest.mark.parametrize(("value", "successes"), [(1.0, 5), (0.0, 0)])
def test_a_unanimous_mean_falls_back_to_wilson(value, successes):
    # Zero spread gives t no width; 5/5 is not certainty.
    assert mean_interval([value] * 5) == wilson(successes, 5)


def test_a_unanimous_weighted_mean_bounds_on_kish_n_and_counts_fixtures():
    # Kish n 36 / 12 = 3.
    interval = mean_interval([1.0] * 4, [3.0, 1.0, 1.0, 1.0])
    assert _bounds(interval) == _bounds(wilson(3, 3))
    assert interval is not None and (interval.n, interval.method) == (4, "wilson")


def test_rate_interval_is_wilson_for_binary_fixtures():
    assert rate_interval([1, 0, 0, 1, 0, 0] * 3 + [0, 0]) == wilson(6, 20)


def test_rate_interval_is_t_for_fractional_fixtures():
    interval = rate_interval([1 / 3, 0.0, 1.0, 2 / 3])
    assert interval is not None and interval.method == "t" and interval.n == 4


def test_rate_interval_on_nothing_is_none():
    assert rate_interval([]) is None


def test_format_is_whole_percent_with_n():
    interval = wilson(6, 20)
    assert interval is not None
    assert interval.format() == "95% CI 15–52%, n=20"


def test_paired_interval_needs_two_fixtures() -> None:
    assert paired_interval([]) is None
    assert paired_interval([0.5]) is None


def test_identical_pairs_are_a_zero_width_interval_at_zero() -> None:
    interval = paired_interval([0.0, 0.0, 0.0])
    assert interval is not None
    assert (interval.low, interval.high, interval.n, interval.method) == (0.0, 0.0, 3, "paired-t")


def test_a_consistent_gap_excludes_zero() -> None:
    interval = paired_interval([0.5, 0.4, 0.6, 0.5, 0.5])
    assert interval is not None
    assert interval.low > 0


def test_noise_straddles_zero_and_is_not_clipped() -> None:
    interval = paired_interval([1.0, -1.0, 1.0, -1.0])
    assert interval is not None
    assert interval.low < -1.0 < 0 < 1.0 < interval.high
