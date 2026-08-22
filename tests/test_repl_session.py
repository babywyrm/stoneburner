"""In-memory REPL session: set, clear, reject unknown keys."""

from __future__ import annotations

import pytest

from atomics.repl.session import SESSION_KEYS, Session, SessionError


def test_new_session_is_empty() -> None:
    session = Session()
    assert session.as_dict() == {
        "provider": None,
        "model": None,
        "effort": None,
        "reasoning_mode": None,
        "host": None,
        "last_job_id": None,
    }


def test_set_writes_a_known_key() -> None:
    session = Session()
    session.set("model", "gpt-oss:20b")
    assert session.model == "gpt-oss:20b"


def test_set_without_value_clears() -> None:
    session = Session(model="x")
    session.set("model", None)
    assert session.model is None


def test_unknown_key_lists_the_five_names() -> None:
    with pytest.raises(SessionError, match="provider") as exc:
        Session().set("temperature", "0.2")
    text = str(exc.value)
    for key in SESSION_KEYS:
        assert key in text
