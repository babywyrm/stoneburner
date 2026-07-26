"""The catalog is the shared vocabulary every fixture draws from."""

from __future__ import annotations

import pytest

from atomics.eval.toolcall.catalog import CATALOG, schemas_for


def test_every_schema_has_the_three_fields_a_dialect_needs():
    for key, schema in CATALOG.items():
        assert schema["name"] == key, f"{key}: name must match its catalog key"
        assert schema["description"], f"{key}: needs a description"
        params = schema["parameters"]
        assert params["type"] == "object"
        assert params["properties"], f"{key}: needs at least one property"


def test_every_declared_required_argument_exists_in_properties():
    for key, schema in CATALOG.items():
        params = schema["parameters"]
        for name in params.get("required", []):
            assert name in params["properties"], f"{key}: required arg {name} not defined"


def test_a_benign_tool_exists_for_the_capability_probe():
    """Without a safe alternative you cannot tell refusal from safe tool choice."""
    assert "get_weather" in CATALOG


def test_schemas_for_returns_them_in_the_requested_order():
    assert [s["name"] for s in schemas_for(("read_file", "get_weather"))] == [
        "read_file",
        "get_weather",
    ]


def test_an_unknown_key_fails_loudly():
    with pytest.raises(KeyError, match="nonexistent_tool"):
        schemas_for(("nonexistent_tool",))


def test_schemas_for_returns_copies_so_a_run_cannot_mutate_the_catalog():
    """Schemas go over the wire to every model; a mutated one silently changes
    what later fixtures were asked."""
    schema = schemas_for(("read_file",))[0]
    schema["description"] = "tampered"
    assert CATALOG["read_file"]["description"] != "tampered"
