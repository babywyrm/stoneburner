"""Cohorts: bands, candidates, placement, judge, and the cohorts command."""

from __future__ import annotations

import pytest

from atomics.eval.batteries import get_battery
from atomics.inventory.cohorts import battery_needs, parameter_band


@pytest.mark.parametrize(
    ("params", "cls", "band"),
    [
        ("3.8B", "mid", "<5B"),
        ("350M", None, "<5B"),
        ("5B", None, "5-15B"),
        ("14.7B", None, "5-15B"),
        ("15B", None, "15-40B"),
        ("40B", None, ">40B"),
        ("1.2T", None, ">40B"),
        (None, "light", "<5B"),
        ("", "mid", "5-15B"),
        ("big", "heavy", "15-40B"),
        (None, None, None),
    ],
)
def test_parameter_band(params, cls, band):
    assert parameter_band(params, cls) == band


def test_battery_needs_tools_only_when_toolcall_runs():
    assert battery_needs(get_battery("desk-pass")) == {"completion", "tools"}
    assert battery_needs(get_battery("red-capability")) == {"completion"}
