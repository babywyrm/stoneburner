"""Tests for the code generation evaluation suite."""

from __future__ import annotations

import pytest

from atomics.eval.codegen import CodegenFixture, CodeTestCase
from atomics.eval.codegen.fixtures import ALL_CODEGEN_FIXTURES
from atomics.eval.codegen.runner import (
    CodegenFixtureResult,
    CodegenRunSummary,
    extract_function,
    run_test_case,
)
from atomics.models import TaskCategory, TaskComplexity, TaskResult, TaskStatus


# ── Fixture collection tests ────────────────────────────────────────────────


def test_all_fixtures_loaded():
    assert len(ALL_CODEGEN_FIXTURES) == 15


def test_fixture_ids_are_unique():
    ids = [f.id for f in ALL_CODEGEN_FIXTURES]
    assert len(ids) == len(set(ids))


def test_fixture_ids_follow_convention():
    for f in ALL_CODEGEN_FIXTURES:
        assert f.id.startswith("cg-"), f"Fixture {f.id} doesn't follow cg-NN pattern"


def test_fixtures_have_test_cases():
    for f in ALL_CODEGEN_FIXTURES:
        assert len(f.test_cases) >= 2, f"Fixture {f.id} has fewer than 2 test cases"


def test_complexity_spread():
    complexities = {f.complexity for f in ALL_CODEGEN_FIXTURES}
    assert TaskComplexity.LIGHT in complexities
    assert TaskComplexity.MODERATE in complexities
    assert TaskComplexity.HEAVY in complexities


def test_all_fixtures_are_python():
    for f in ALL_CODEGEN_FIXTURES:
        assert f.language == "python"


def test_fixtures_have_function_names():
    for f in ALL_CODEGEN_FIXTURES:
        assert f.function_name, f"Fixture {f.id} has no function_name"
        assert f.signature.startswith("def "), f"Fixture {f.id} signature doesn't start with 'def'"


# ── Function extraction tests ───────────────────────────────────────────────


def test_extract_from_code_block():
    response = '```python\ndef fizzbuzz(n):\n    return str(n)\n```'
    code = extract_function(response, "fizzbuzz")
    assert code is not None
    assert "def fizzbuzz" in code


def test_extract_from_raw_function():
    response = 'def fizzbuzz(n):\n    return str(n)\n'
    code = extract_function(response, "fizzbuzz")
    assert code is not None
    assert "def fizzbuzz" in code


def test_extract_with_prose_before():
    response = 'Here is the implementation:\n\n```python\ndef add(a, b):\n    return a + b\n```\n\nThis works by...'
    code = extract_function(response, "add")
    assert code is not None
    assert "def add" in code
    assert "This works" not in code


def test_extract_returns_none_for_missing():
    response = "I don't know how to write that function."
    code = extract_function(response, "fizzbuzz")
    assert code is None


def test_extract_code_block_without_lang():
    response = '```\ndef count(n):\n    return n\n```'
    code = extract_function(response, "count")
    assert code is not None


# ── Test execution tests ────────────────────────────────────────────────────


def test_run_test_case_pass():
    code = "def add(a, b):\n    return a + b\n"
    fixture = CodegenFixture(
        id="test", complexity=TaskComplexity.LIGHT, language="python",
        function_name="add", description="", signature="def add(a, b):",
        test_cases=[],
    )
    tc = CodeTestCase([2, 3], 5)
    passed, detail = run_test_case(code, "add", tc, fixture)
    assert passed is True
    assert detail == "OK"


def test_run_test_case_fail():
    code = "def add(a, b):\n    return a - b\n"
    fixture = CodegenFixture(
        id="test", complexity=TaskComplexity.LIGHT, language="python",
        function_name="add", description="", signature="def add(a, b):",
        test_cases=[],
    )
    tc = CodeTestCase([2, 3], 5)
    passed, detail = run_test_case(code, "add", tc, fixture)
    assert passed is False
    assert "Expected" in detail


def test_run_test_case_compile_error():
    code = "def add(a, b)\n    return a + b\n"
    fixture = CodegenFixture(
        id="test", complexity=TaskComplexity.LIGHT, language="python",
        function_name="add", description="", signature="def add(a, b):",
        test_cases=[],
    )
    tc = CodeTestCase([2, 3], 5)
    passed, detail = run_test_case(code, "add", tc, fixture)
    assert passed is False
    assert "Compilation" in detail or "SyntaxError" in detail


def test_run_test_case_runtime_error():
    code = "def divide(a, b):\n    return a / b\n"
    fixture = CodegenFixture(
        id="test", complexity=TaskComplexity.LIGHT, language="python",
        function_name="divide", description="", signature="def divide(a, b):",
        test_cases=[],
    )
    tc = CodeTestCase([1, 0], 0)
    passed, detail = run_test_case(code, "divide", tc, fixture)
    assert passed is False
    assert "Runtime error" in detail


def test_run_test_case_function_not_found():
    code = "x = 42\n"
    fixture = CodegenFixture(
        id="test", complexity=TaskComplexity.LIGHT, language="python",
        function_name="add", description="", signature="def add(a, b):",
        test_cases=[],
    )
    tc = CodeTestCase([2, 3], 5)
    passed, detail = run_test_case(code, "add", tc, fixture)
    assert passed is False
    assert "not found" in detail


# ── Fixtures correctness smoke test ─────────────────────────────────────────


def test_fizzbuzz_fixture_has_valid_expectations():
    """Sanity-check that the FizzBuzz test cases are internally consistent."""
    fixture = next(f for f in ALL_CODEGEN_FIXTURES if f.id == "cg-01")
    reference = """
def fizzbuzz(n):
    if n % 15 == 0: return "FizzBuzz"
    if n % 3 == 0: return "Fizz"
    if n % 5 == 0: return "Buzz"
    return str(n)
"""
    for tc in fixture.test_cases:
        passed, detail = run_test_case(reference, "fizzbuzz", tc, fixture)
        assert passed, f"Reference fizzbuzz failed on {tc.input_args}: {detail}"


def test_palindrome_fixture_has_valid_expectations():
    fixture = next(f for f in ALL_CODEGEN_FIXTURES if f.id == "cg-02")
    reference = """
def is_palindrome(s):
    cleaned = s.replace(" ", "").lower()
    return cleaned == cleaned[::-1]
"""
    for tc in fixture.test_cases:
        passed, detail = run_test_case(reference, "is_palindrome", tc, fixture)
        assert passed, f"Reference is_palindrome failed on {tc.input_args}: {detail}"


def test_fibonacci_fixture_has_valid_expectations():
    fixture = next(f for f in ALL_CODEGEN_FIXTURES if f.id == "cg-03")
    reference = """
def fibonacci(n):
    if n <= 0: return 0
    if n == 1: return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
"""
    for tc in fixture.test_cases:
        passed, detail = run_test_case(reference, "fibonacci", tc, fixture)
        assert passed, f"Reference fibonacci failed on {tc.input_args}: {detail}"


# ── Summary tests ────────────────────────────────────────────────────────────


def test_summary_overall_pass_rate():
    from datetime import UTC, datetime

    fr1 = CodegenFixtureResult(
        fixture=ALL_CODEGEN_FIXTURES[0],
        task_result=TaskResult(
            run_id="t", category=TaskCategory.GENERAL_QA, task_name="cg-01",
            provider="m", model="m", status=TaskStatus.SUCCESS,
            total_tokens=100, estimated_cost_usd=0.01,
            started_at=datetime.now(UTC), completed_at=datetime.now(UTC),
        ),
        tests_passed=5, tests_total=6, pass_rate=5/6,
        extracted_code="x", test_details=[],
    )
    summary = CodegenRunSummary(
        run_id="t", provider="m", model="m",
        started_at=datetime.now(UTC), completed_at=datetime.now(UTC),
        fixture_results=[fr1],
    )
    assert summary.overall_pass_rate == round(5/6, 3)
    assert summary.fixtures_fully_correct == 0


def test_summary_to_dict():
    from datetime import UTC, datetime

    summary = CodegenRunSummary(
        run_id="t", provider="m", model="m",
        started_at=datetime.now(UTC), completed_at=datetime.now(UTC),
    )
    d = summary.to_dict()
    assert "overall_pass_rate" in d
    assert "fixtures_fully_correct" in d
    assert d["total_fixtures"] == 0


# ── CLI tests ────────────────────────────────────────────────────────────────


def test_cli_codegen_help():
    from click.testing import CliRunner
    from atomics.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["codegen", "--help"])
    assert result.exit_code == 0
    assert "functional correctness" in result.output.lower() or "Code generation" in result.output
