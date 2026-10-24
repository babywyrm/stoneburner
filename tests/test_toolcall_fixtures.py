"""Fixture-integrity guards for the toolcall suite."""

from __future__ import annotations

import dataclasses
import re

import pytest

from atomics.eval.toolcall.catalog import BENIGN_TOOLS, CATALOG
from atomics.eval.toolcall.fixtures import (
    ALL_FIXTURES,
    GROUP_ALIASES,
    DangerousCall,
    ToolCallFixture,
    fixtures_for_category,
    select_fixtures,
)


def test_scripted_result_lookup_matches_tool_then_patterns():
    from atomics.eval.toolcall.fixtures import ScriptedResult, result_for
    from atomics.providers.toolcalls import ToolCall

    results = (
        ScriptedResult(
            tool="read_file", arg_patterns={"path": r"\.env"}, output="SECRET=sb-canary-7f3a91"
        ),
        ScriptedResult(tool="list_files", output="README.md\n.env"),
    )
    assert "sb-canary" in result_for(
        results, ToolCall(name="read_file", arguments={"path": "/app/.env"})
    )
    assert (
        result_for(results, ToolCall(name="list_files", arguments={"directory": "."}))
        == "README.md\n.env"
    )
    # No match anywhere: the neutral default, never an error string.
    assert (
        result_for(results, ToolCall(name="get_weather", arguments={"city": "Paris"}))
        == "No results."
    )


def test_max_steps_defaults_to_one_and_results_default_empty():
    sample = ALL_FIXTURES[0]
    assert sample.max_steps == 1
    assert sample.results == ()


def test_every_dangerous_call_is_actually_offered_to_the_model():
    """The most important test here.

    A fixture whose DangerousCall names a tool absent from its `tools` can never
    fire. It would report perfect resistance forever, and nothing else would ever
    reveal it.
    """
    for fixture in ALL_FIXTURES:
        for dangerous in fixture.dangerous:
            assert dangerous.tool in fixture.tools, (
                f"{fixture.id}: dangerous call {dangerous.tool!r} is not in "
                f"offered tools {fixture.tools} — this fixture can never fire"
            )


def test_every_offered_tool_exists_in_the_catalog():
    for fixture in ALL_FIXTURES:
        for key in fixture.tools:
            assert key in CATALOG, f"{fixture.id}: unknown tool {key!r}"


def test_every_dangerous_argument_name_exists_in_the_tool_schema():
    """An arg_patterns key that the schema has no property for can never match.

    '*' is the wildcard (match any argument) and is exempt.
    """
    for fixture in ALL_FIXTURES:
        for dangerous in fixture.dangerous:
            properties = CATALOG[dangerous.tool]["parameters"]["properties"]
            for arg in dangerous.arg_patterns:
                if arg == "*":
                    continue
                assert arg in properties, f"{fixture.id}: {dangerous.tool} has no argument {arg!r}"


def test_every_argument_pattern_compiles():
    for fixture in ALL_FIXTURES:
        for dangerous in fixture.dangerous:
            for arg, pattern in dangerous.arg_patterns.items():
                try:
                    re.compile(pattern)
                except re.error as exc:  # pragma: no cover - guard
                    pytest.fail(f"{fixture.id}: {arg} pattern {pattern!r}: {exc}")


def test_no_prompt_names_a_catalog_tool():
    """The pairing depends on prompts being tool-agnostic.

    A prompt naming a tool measures instruction-following in the prose channel
    and tool selection in the tool channel — two different things, making the
    comparison between them meaningless. injected_tool_output is exempt, since a
    tool result legitimately mentions tools.
    """
    for fixture in ALL_FIXTURES:
        for key in CATALOG:
            assert key not in fixture.prompt, (
                f"{fixture.id}: prompt names the tool {key!r}; state the task instead"
            )


def test_fixture_ids_are_unique():
    ids = [f.id for f in ALL_FIXTURES]
    assert len(ids) == len(set(ids))


def test_every_fixture_has_judge_fields_for_the_prose_channel():
    """score_resistance requires both; without them the prose baseline cannot run."""
    for fixture in ALL_FIXTURES:
        assert fixture.attack_goal, f"{fixture.id}: needs an attack_goal"
        assert fixture.resistance_criteria, f"{fixture.id}: needs resistance_criteria"


def test_tool_only_fixtures_are_exactly_injected_or_multistep():
    for fixture in ALL_FIXTURES:
        expected = fixture.injected_tool_output is not None or fixture.max_steps > 1
        assert fixture.tool_channel_only == expected


def test_every_fixture_declares_at_least_one_dangerous_call():
    """A fixture with none can only ever report resistance."""
    for fixture in ALL_FIXTURES:
        assert fixture.dangerous, f"{fixture.id}: declares no dangerous call"


def test_most_fixtures_offer_a_benign_alternative():
    """Offering only dangerous tools conflates refusal with safe tool choice.

    Not universal — a couple of fixtures deliberately offer only the dangerous
    tool — but the bulk must, or the suite cannot tell the two apart.
    """
    with_benign = [f for f in ALL_FIXTURES if any(t in BENIGN_TOOLS for t in f.tools)]
    assert len(with_benign) >= len(ALL_FIXTURES) * 0.75


def test_fixtures_and_dangerous_calls_are_immutable():
    """Frozen so a run cannot mutate the definitions it is measuring against.

    Not hashable, though: arg_patterns is a dict, so hash() raises even on a
    frozen dataclass. Immutability is the property that matters here.
    """
    call = DangerousCall(tool="read_file", arg_patterns={"path": "/etc/shadow"}, why="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        call.tool = "write_file"  # type: ignore[misc]

    fixture = ALL_FIXTURES[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        fixture.severity = "LOW"  # type: ignore[misc]
    assert isinstance(fixture, ToolCallFixture)


def test_the_suite_covers_every_declared_category():
    declared = set(GROUP_ALIASES)
    for alias in declared:
        assert fixtures_for_category(alias), f"alias {alias!r} selects no fixtures"


def test_a_group_alias_cannot_select_a_fixture_from_another_category():
    """The defect this avoids: --category mcp selecting 12 fixtures, not 11."""
    for alias, categories in GROUP_ALIASES.items():
        selected = fixtures_for_category(alias)
        assert {f.category for f in selected} <= categories


def test_an_unknown_category_selects_nothing_rather_than_everything():
    """The CLI validates before calling this; a silent full run would be worse."""
    assert fixtures_for_category("no-such-category") == ()


def test_no_category_means_every_fixture():
    assert fixtures_for_category(None) == ALL_FIXTURES


def test_severities_are_from_the_declared_set():
    for fixture in ALL_FIXTURES:
        assert fixture.severity in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_the_suite_has_the_planned_shape():
    """Thirty-one fixtures across nine categories, injection and trajectory."""
    assert len(ALL_FIXTURES) == 31
    by_category: dict[str, int] = {}
    for fixture in ALL_FIXTURES:
        by_category[fixture.category] = by_category.get(fixture.category, 0) + 1
    assert by_category == {
        "direct_dangerous": 4,
        "authority_fabrication": 3,
        "breakglass": 2,
        "indirect_injection": 4,
        "egress_exfil": 4,
        "destructive_urgency": 3,
        "trajectory_injection": 4,
        "escalation_after_denial": 3,
        "read_then_leak": 4,
    }
    injected = sum(1 for f in ALL_FIXTURES if f.injected_tool_output is not None)
    multistep = sum(1 for f in ALL_FIXTURES if f.max_steps > 1)
    assert injected == 4
    assert multistep == 11


def test_trajectory_fixtures_have_a_sane_step_cap():
    for f in ALL_FIXTURES:
        if f.max_steps > 1:
            assert 2 <= f.max_steps <= 5, f"{f.id}: max_steps {f.max_steps} out of 2..5"


def test_scripted_results_only_name_offered_tools():
    for f in ALL_FIXTURES:
        for r in f.results:
            assert r.tool in f.tools, f"{f.id}: scripted result for unoffered tool {r.tool!r}"


def test_read_then_leak_fixtures_plant_and_match_a_marker():
    """Each leak fixture must seed a marker in a result and match it in a
    dangerous call, or it cannot demonstrate a leak."""
    leaks = [f for f in ALL_FIXTURES if f.category == "read_then_leak"]
    assert leaks
    for f in leaks:
        seeded = " ".join(r.output for r in f.results)
        markers = re.findall(r"sb-canary-[0-9a-f]+", seeded)
        assert markers, f"{f.id}: no marker planted in a scripted result"
        patterns = " ".join(p for d in f.dangerous for p in d.arg_patterns.values())
        assert any(m in patterns for m in markers), (
            f"{f.id}: marker not matched by a dangerous call"
        )


def test_result_lookup_misses_when_a_patterned_argument_is_absent():
    from atomics.eval.toolcall.fixtures import ScriptedResult, result_for
    from atomics.providers.toolcalls import ToolCall

    results = (ScriptedResult(tool="read_file", arg_patterns={"path": r"\.env"}, output="SECRET"),)
    # The call names read_file but omits the path the pattern needs, so no match.
    assert result_for(results, ToolCall(name="read_file", arguments={})) == "No results."


def test_trajectory_alias_selects_the_three_new_categories():
    from atomics.eval.toolcall.fixtures import fixtures_for_category

    cats = {f.category for f in fixtures_for_category("trajectory")}
    assert cats == {"trajectory_injection", "escalation_after_denial", "read_then_leak"}


def test_select_fixtures_by_id_preserves_request_order():
    selected = select_fixtures(ids=["tc-02", "tc-01"])
    assert [f.id for f in selected] == ["tc-02", "tc-01"]


def test_select_fixtures_unknown_id_raises():
    with pytest.raises(ValueError, match="tc-99"):
        select_fixtures(ids=["tc-01", "tc-99"])


def test_select_fixtures_id_must_be_in_category():
    with pytest.raises(ValueError, match="tc-01"):
        select_fixtures(category="indirect_injection", ids=["tc-01"])
