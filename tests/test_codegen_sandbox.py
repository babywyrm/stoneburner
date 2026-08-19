"""Containment tests for the codegen execution sandbox.

Model-generated code is untrusted and the codegen suite is reachable over HTTP,
so these assert the boundary itself rather than just the happy path.
"""

from __future__ import annotations

import time

import pytest

from atomics.eval.codegen import sandbox


def test_a_correct_function_returns_its_value():
    outcome = sandbox.execute("def add(a, b):\n    return a + b", "add", (2, 3))
    assert outcome.status == "ok"
    assert outcome.result == 5


def test_non_json_types_survive_the_round_trip():
    outcome = sandbox.execute("def f():\n    return (1, {2}, [3])", "f", ())
    assert outcome.status == "ok"
    assert outcome.result == (1, {2}, [3])


def test_syntax_errors_are_reported_not_raised():
    outcome = sandbox.execute("def broken(:", "broken", ())
    assert outcome.status == "compile_error"
    assert "SyntaxError" in outcome.detail


def test_a_missing_function_is_distinguished_from_a_crash():
    outcome = sandbox.execute("def other():\n    return 1", "wanted", ())
    assert outcome.status == "missing_function"


def test_runtime_errors_carry_the_exception_type():
    outcome = sandbox.execute("def f():\n    return 1 / 0", "f", ())
    assert outcome.status == "runtime_error"
    assert "ZeroDivisionError" in outcome.detail


def test_provider_credentials_are_not_visible_to_generated_code(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-canary")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-canary")
    outcome = sandbox.execute(
        "import os\ndef f():\n    return sorted(os.environ.values())", "f", ()
    )
    assert outcome.status == "ok"
    assert not any("canary" in value for value in outcome.result)


def test_an_infinite_loop_inside_the_function_is_killed():
    started = time.monotonic()
    outcome = sandbox.execute("def f():\n    while True:\n        pass", "f", (), timeout_seconds=2)
    assert outcome.status == "timeout"
    assert time.monotonic() - started < 10


def test_module_level_code_is_covered_by_the_timeout():
    """The in-process version armed its alarm after exec, leaving this unguarded."""
    started = time.monotonic()
    outcome = sandbox.execute(
        "while True:\n    pass\ndef f():\n    return 1", "f", (), timeout_seconds=2
    )
    assert outcome.status == "timeout"
    assert time.monotonic() - started < 10


def test_generated_code_cannot_open_a_network_connection():
    outcome = sandbox.execute(
        "import socket\n"
        "def f():\n"
        "    return socket.create_connection(('192.0.2.1', 80), timeout=1)",
        "f",
        (),
    )
    assert outcome.status == "runtime_error"
    assert "network access is disabled" in outcome.detail


def test_the_child_runs_outside_the_project_directory(tmp_path):
    outcome = sandbox.execute("import os\ndef f():\n    return os.getcwd()", "f", ())
    assert outcome.status == "ok"
    assert "stoneburner" not in outcome.result


def test_killing_the_interpreter_does_not_take_the_parent_with_it():
    outcome = sandbox.execute("import os\ndef f():\n    os._exit(1)", "f", ())
    assert outcome.status == "crashed"


def test_stdout_from_generated_code_does_not_corrupt_the_result():
    outcome = sandbox.execute(
        "print('module noise')\ndef f():\n    print('call noise')\n    return 42", "f", ()
    )
    assert outcome.status == "ok"
    assert outcome.result == 42


@pytest.mark.parametrize(
    "args,expected",
    [((), None), ((1,), 1), (([1, 2],), [1, 2])],
)
def test_arguments_are_passed_through_faithfully(args, expected):
    outcome = sandbox.execute("def f(x=None):\n    return x", "f", args)
    assert outcome.status == "ok"
    assert outcome.result == expected
