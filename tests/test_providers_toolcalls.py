"""Unit tests for the provider-layer tool-call representation."""

from __future__ import annotations

from atomics.providers.toolcalls import ToolCall, parse_arguments


def test_arguments_parse_from_a_json_string():
    """Every dialect delivers arguments as a JSON string, not a dict."""
    args, malformed = parse_arguments('{"path": "/etc/shadow"}')
    assert args == {"path": "/etc/shadow"}
    assert malformed is False


def test_malformed_arguments_are_flagged_not_raised():
    """A model emitting broken JSON is a finding, not a crash."""
    args, malformed = parse_arguments('{"path": ')
    assert args == {}
    assert malformed is True


def test_a_dict_is_accepted_as_already_parsed():
    """Anthropic hands back a parsed object rather than a string."""
    args, malformed = parse_arguments({"path": "/etc/shadow"})
    assert args == {"path": "/etc/shadow"}
    assert malformed is False


def test_non_object_json_counts_as_malformed():
    """A bare list or scalar cannot be argument bindings."""
    args, malformed = parse_arguments("[1, 2, 3]")
    assert args == {}
    assert malformed is True


def test_tool_call_carries_name_arguments_and_malformed_flag():
    call = ToolCall(name="read_file", arguments={"path": "/etc/shadow"})
    assert call.name == "read_file"
    assert call.arguments["path"] == "/etc/shadow"
    assert call.malformed is False
