"""Cohorts: bands, candidates, placement, judge, and the cohorts command."""

from __future__ import annotations

import pytest

from atomics.eval.batteries import get_battery
from atomics.inventory.cohorts import CohortError, battery_needs, form_cohorts, parameter_band


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


def _model(
    name,
    host="laptop",
    *,
    params="8B",
    cls="mid",
    digest=None,
    tools=True,
    completion=True,
    size=5_000_000_000,
    recommended=None,
    tps=None,
    judge=None,
):
    probe = None
    if recommended or tps:
        probe = {"off": {"tokens_per_second": tps}, "recommended": recommended}
    return {
        "name": name,
        "host": host,
        "digest": digest or f"sha-{name}",
        "size_bytes": size,
        "parameter_size": params,
        "model_class": cls,
        "evaluable": completion,
        "capabilities": {
            "completion": {"value": completion, "source": "declared"},
            "tools": {"value": tools, "source": "probe"},
        },
        "probe": probe,
        "judge": judge,
    }


def _inv(*models, hosts=("laptop",)):
    return {
        "schema": 1,
        "taken_at": "t",
        "hosts": [{"label": h} for h in hosts],
        "models": list(models),
    }


DESK = get_battery("desk-pass")


def _names(result):
    return [[m.name for m in c.members] for c in result.cohorts]


def _excluded(result):
    return {(x.name, x.reason) for x in result.excluded}


def test_groups_by_band_and_excludes_with_reasons():
    inv = _inv(
        _model("a:8b"),
        _model("b:7b", params="7B"),
        _model("c:3b", params="3B"),
        _model("d:8b", tools=False),
        _model("e:8b", completion=False),
    )
    result = form_cohorts(inv, DESK)
    assert _names(result) == [["a:8b", "b:7b"]]
    assert _excluded(result) == {
        ("c:3b", "alone in band"),
        ("d:8b", "lacks tools"),
        ("e:8b", "not evaluable"),
    }


def test_unknown_capability_counts_as_missing():
    m = _model("a:8b")
    m["capabilities"]["tools"] = {"value": None, "source": "unknown"}
    result = form_cohorts(_inv(m, _model("b:8b")), DESK)
    assert ("a:8b", "lacks tools") in _excluded(result)


def test_patterns_filter_and_report():
    inv = _inv(_model("gemma4:8b"), _model("gemma4:9b"), _model("qwen3:8b"))
    result = form_cohorts(inv, DESK, patterns=("gemma4*",))
    assert _names(result) == [["gemma4:8b", "gemma4:9b"]]
    assert _excluded(result) == {("qwen3:8b", "filtered by -m")}


def test_same_digest_on_two_hosts_is_one_member():
    inv = _inv(_model("a:8b"), _model("a:8b", "beefy"), _model("b:8b"), hosts=("laptop", "beefy"))
    [cohort] = form_cohorts(inv, DESK).cohorts
    assert [(m.name, m.hosts) for m in cohort.members] == [
        ("a:8b", ("laptop", "beefy")),
        ("b:8b", ("laptop",)),
    ]


def test_copy_that_fails_on_one_host_is_not_listed_excluded():
    inv = _inv(
        _model("a:8b", tools=False),
        _model("a:8b", "beefy"),
        _model("b:8b"),
        hosts=("laptop", "beefy"),
    )
    result = form_cohorts(inv, DESK)
    assert _names(result) == [["a:8b", "b:8b"]]
    assert result.cohorts[0].members[0].hosts == ("beefy",)
    assert result.excluded == []


def test_digest_mismatch_splits_into_labelled_members():
    inv = _inv(
        _model("a:8b", digest="x"), _model("a:8b", "beefy", digest="y"), hosts=("laptop", "beefy")
    )
    assert _names(form_cohorts(inv, DESK)) == [["a:8b@beefy", "a:8b@laptop"]]


def test_max_members_splits_band_evenly():
    inv = _inv(*(_model(f"m{i}:8b") for i in range(4)))
    assert _names(form_cohorts(inv, DESK, max_members=2)) == [
        ["m0:8b", "m2:8b"],
        ["m1:8b", "m3:8b"],
    ]


def test_thinking_follows_probe_recommendation():
    inv = _inv(
        _model("a:8b", recommended="--thinking"),
        _model("b:8b", recommended="--no-thinking"),
        _model("c:8b"),
    )
    [cohort] = form_cohorts(inv, DESK).cohorts
    assert [m.thinking for m in cohort.members] == [True, False, False]


def test_model_without_size_is_excluded():
    inv = _inv(_model("a:8b"), _model("b:8b"), _model("c", params=None, cls=None))
    assert ("c", "no size") in _excluded(form_cohorts(inv, DESK))


def test_rejects_other_schema():
    with pytest.raises(CohortError, match="schema"):
        form_cohorts({"schema": 2, "models": []}, DESK)
