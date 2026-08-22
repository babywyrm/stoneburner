"""Local REPL verbs do not touch the API client."""

from __future__ import annotations

import json

from atomics.repl.dispatch import handle_line
from atomics.repl.session import Session


class _Boom:
    def __getattr__(self, name: str):
        raise AssertionError(f"local verb must not call client.{name}")


def test_show_prints_session_json() -> None:
    session = Session(provider="ollama", model="gpt-oss:20b")
    result = handle_line("show", session=session, client=_Boom())  # type: ignore[arg-type]
    assert json.loads(result.stdout) == session.as_dict()
    assert result.exit_loop is False


def test_set_key_value() -> None:
    session = Session()
    handle_line("set effort high", session=session, client=_Boom())  # type: ignore[arg-type]
    assert session.effort == "high"


def test_set_key_clears() -> None:
    session = Session(model="x")
    handle_line("set model", session=session, client=_Boom())  # type: ignore[arg-type]
    assert session.model is None


def test_unknown_verb_stays_in_the_prompt() -> None:
    result = handle_line("frobnicate", session=Session(), client=_Boom())  # type: ignore[arg-type]
    assert result.exit_loop is False
    assert "help" in result.stderr.lower()
    assert "frobnicate" in result.stderr


def test_exit_asks_the_loop_to_stop() -> None:
    result = handle_line("exit", session=Session(), client=_Boom())  # type: ignore[arg-type]
    assert result.exit_loop is True
    assert result.exit_code == 0


def test_help_lists_submit_eval() -> None:
    result = handle_line("help", session=Session(), client=_Boom())  # type: ignore[arg-type]
    assert "submit_eval" in result.stdout
    assert "wait" in result.stdout
