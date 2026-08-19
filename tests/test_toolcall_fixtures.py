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
)


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
    """An arg_patterns key that the schema has no property for can never match."""
    for fixture in ALL_FIXTURES:
        for dangerous in fixture.dangerous:
            properties = CATALOG[dangerous.tool]["parameters"]["properties"]
            for arg in dangerous.arg_patterns:
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


def test_tool_only_fixtures_are_exactly_those_with_injected_output():
    for fixture in ALL_FIXTURES:
        assert fixture.tool_channel_only == (fixture.injected_tool_output is not None)


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
    """Twenty fixtures across six categories, with indirect injection present."""
    assert len(ALL_FIXTURES) == 20
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
    }
    assert sum(1 for f in ALL_FIXTURES if f.tool_channel_only) == 4
