"""shlex line parser: verb, one positional, --flag value pairs."""

from __future__ import annotations

import pytest

from atomics.repl.parse import ParseError, parse_line


def test_blank_line_is_none() -> None:
    assert parse_line("   ") is None


def test_verb_only() -> None:
    parsed = parse_line("show")
    assert parsed is not None
    assert parsed.verb == "show"
    assert parsed.args == ()
    assert parsed.flags == {}


def test_one_positional() -> None:
    parsed = parse_line("get_job 3f2a")
    assert parsed is not None
    assert parsed.verb == "get_job"
    assert parsed.args == ("3f2a",)


def test_flag_value_pairs() -> None:
    parsed = parse_line("submit_eval --suite toolcall --provider ollama")
    assert parsed is not None
    assert parsed.verb == "submit_eval"
    assert parsed.flags == {"suite": "toolcall", "provider": "ollama"}


def test_equals_form() -> None:
    parsed = parse_line("submit_eval --suite=toolcall")
    assert parsed is not None
    assert parsed.flags == {"suite": "toolcall"}


def test_quoted_value() -> None:
    parsed = parse_line('set model "gpt oss"')
    assert parsed is not None
    assert parsed.verb == "set"
    assert parsed.args == ("model", "gpt oss")


def test_unknown_bare_dash_is_an_error() -> None:
    with pytest.raises(ParseError, match="flag"):
        parse_line("health -x")


def test_flag_without_value_is_an_error() -> None:
    with pytest.raises(ParseError, match="value"):
        parse_line("submit_eval --suite")
