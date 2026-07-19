"""Codegen eval runner — generates code, extracts functions, runs tests."""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from atomics.eval.codegen import CodegenFixture, CodeTestCase
from atomics.eval.codegen.fixtures import ALL_CODEGEN_FIXTURES
from atomics.models import TaskCategory, TaskResult, TaskStatus
from atomics.providers.base import BaseProvider
from atomics.validation import sanitize_error

logger = logging.getLogger("atomics.eval.codegen.runner")

_CODEGEN_SYSTEM = (
    "You are a Python programmer. Write ONLY the function implementation — "
    "no explanations, no examples, no test code. Output valid Python code only."
)

_CODEGEN_PROMPT = """\
Implement the following Python function:

{signature}

Description: {description}

Write ONLY the function definition. Do not include any other code, imports should be inside the function if needed, no examples, no tests, no explanations.
"""


def extract_function(response: str, function_name: str) -> str | None:
    """Extract a Python function from model output.

    Handles code blocks (```python ... ```) and raw function definitions.
    """
    code_block = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()

    func_match = re.search(
        rf"(def\s+{re.escape(function_name)}\s*\(.*?\).*?:.*?)(?:\n(?=\S)|\Z)",
        response,
        re.DOTALL,
    )
    if func_match:
        return func_match.group(1).strip()

    if f"def {function_name}" in response:
        lines = response.split("\n")
        func_lines: list[str] = []
        capturing = False
        for line in lines:
            if line.strip().startswith(f"def {function_name}"):
                capturing = True
            if capturing:
                func_lines.append(line)
                if func_lines and line.strip() and not line[0].isspace() and len(func_lines) > 1:
                    break
        if func_lines:
            return "\n".join(func_lines).strip()

    return None


def _compare_output(actual: object, expected: object, fixture: CodegenFixture) -> bool:
    """Compare actual output to expected, with special handling for some fixtures."""
    if fixture.id == "cg-08":
        if isinstance(actual, list) and isinstance(expected, list):
            return sorted(sorted(g) for g in actual) == sorted(sorted(g) for g in expected)
    if fixture.id == "cg-15":
        if isinstance(actual, list) and isinstance(expected, list):
            if not actual and not expected:
                return True
            if not actual or not expected:
                return len(actual) == len(expected)
            return _is_valid_topo_order(actual, fixture)
    return actual == expected


def _is_valid_topo_order(order: list[int], fixture: CodegenFixture) -> bool:
    """Verify a topological ordering is valid for the fixture's edges."""
    if not fixture.test_cases:
        return True
    tc = fixture.test_cases[0]
    n = tc.input_args[0]
    edges = tc.input_args[1]
    if len(order) != n:
        return False
    pos = {node: i for i, node in enumerate(order)}
    return all(pos.get(u, -1) < pos.get(v, -1) for u, v in edges)


def run_test_case(
    code: str,
    function_name: str,
    test_case: CodeTestCase,
    fixture: CodegenFixture,
    *,
    timeout_seconds: float = 5.0,
) -> tuple[bool, str]:
    """Execute a test case against extracted code. Returns (passed, detail)."""
    namespace: dict = {}
    try:
        exec(code, namespace)  # noqa: S102
    except Exception as exc:
        return False, f"Compilation error: {type(exc).__name__}: {exc}"

    func = namespace.get(function_name)
    if func is None:
        return False, f"Function '{function_name}' not found in generated code"

    try:
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError("Execution timed out")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(timeout_seconds))
        try:
            result = func(*test_case.input_args)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    except TimeoutError:
        return False, "Execution timed out"
    except Exception as exc:
        return False, f"Runtime error: {type(exc).__name__}: {exc}"

    if _compare_output(result, test_case.expected_output, fixture):
        return True, "OK"
    return False, f"Expected {test_case.expected_output!r}, got {result!r}"


@dataclass
class CodegenFixtureResult:
    fixture: CodegenFixture
    task_result: TaskResult
    tests_passed: int
    tests_total: int
    pass_rate: float
    extracted_code: str | None
    test_details: list[tuple[bool, str]]


@dataclass
class CodegenRunSummary:
    run_id: str
    provider: str
    model: str
    started_at: datetime
    completed_at: datetime
    fixture_results: list[CodegenFixtureResult] = field(default_factory=list)

    @property
    def overall_pass_rate(self) -> float | None:
        total_passed = sum(r.tests_passed for r in self.fixture_results)
        total_tests = sum(r.tests_total for r in self.fixture_results)
        return round(total_passed / total_tests, 3) if total_tests > 0 else None

    @property
    def fixtures_fully_correct(self) -> int:
        return sum(1 for r in self.fixture_results if r.pass_rate == 1.0)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.task_result.estimated_cost_usd for r in self.fixture_results)

    @property
    def avg_latency_ms(self) -> float:
        lats = [r.task_result.latency_ms for r in self.fixture_results if r.task_result.latency_ms]
        return round(sum(lats) / len(lats), 1) if lats else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(r.task_result.total_tokens for r in self.fixture_results)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "model": self.model,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "overall_pass_rate": self.overall_pass_rate,
            "fixtures_fully_correct": self.fixtures_fully_correct,
            "total_fixtures": len(self.fixture_results),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_tokens": self.total_tokens,
            "avg_latency_ms": self.avg_latency_ms,
            "fixtures": [
                {
                    "id": r.fixture.id,
                    "complexity": r.fixture.complexity.value,
                    "function": r.fixture.function_name,
                    "tests_passed": r.tests_passed,
                    "tests_total": r.tests_total,
                    "pass_rate": r.pass_rate,
                    "latency_ms": r.task_result.latency_ms,
                    "tokens": r.task_result.total_tokens,
                    "test_details": [
                        {"passed": p, "detail": d} for p, d in r.test_details
                    ],
                }
                for r in self.fixture_results
            ],
        }


async def run_codegen(
    provider: BaseProvider,
    *,
    model: str | None = None,
    run_id: str | None = None,
    on_fixture_done: Callable[[CodegenFixtureResult], None] | None = None,
    thinking: bool | None = None,
    thinking_budget: int | None = None,
    fixtures: list[CodegenFixture] | None = None,
) -> CodegenRunSummary:
    """Run the code generation evaluation suite."""
    effective_run_id = run_id or uuid.uuid4().hex[:12]
    selected = fixtures or ALL_CODEGEN_FIXTURES
    started = datetime.now(UTC)

    results: list[CodegenFixtureResult] = []

    for fixture in selected:
        prompt = _CODEGEN_PROMPT.format(
            signature=fixture.signature,
            description=fixture.description,
        )
        task_started = datetime.now(UTC)

        gen_kwargs: dict = {
            "system": _CODEGEN_SYSTEM,
            "model": model,
            "max_tokens": fixture.max_output_tokens,
        }
        if thinking is not None:
            gen_kwargs["thinking"] = thinking
        if thinking_budget is not None:
            gen_kwargs["thinking_budget"] = thinking_budget

        try:
            resp = await provider.generate(prompt, **gen_kwargs)
            response_text = resp.text
            tr = TaskResult(
                task_id=uuid.uuid4().hex[:12],
                run_id=effective_run_id,
                category=TaskCategory.GENERAL_QA,
                task_name=fixture.id,
                provider=provider.name,
                model=resp.model or model or "",
                status=TaskStatus.SUCCESS,
                prompt=prompt,
                response=response_text,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                total_tokens=resp.total_tokens,
                latency_ms=resp.latency_ms,
                estimated_cost_usd=resp.estimated_cost_usd,
                started_at=task_started,
                completed_at=datetime.now(UTC),
            )
        except Exception as exc:
            tr = TaskResult(
                task_id=uuid.uuid4().hex[:12],
                run_id=effective_run_id,
                category=TaskCategory.GENERAL_QA,
                task_name=fixture.id,
                provider=provider.name,
                model=model or "",
                status=TaskStatus.FAILED,
                prompt=prompt,
                error_class=type(exc).__name__,
                error_message=sanitize_error(exc),
                started_at=task_started,
                completed_at=datetime.now(UTC),
            )
            fr = CodegenFixtureResult(
                fixture=fixture, task_result=tr,
                tests_passed=0, tests_total=len(fixture.test_cases),
                pass_rate=0.0, extracted_code=None, test_details=[],
            )
            results.append(fr)
            if on_fixture_done:
                on_fixture_done(fr)
            continue

        code = extract_function(response_text, fixture.function_name)
        test_details: list[tuple[bool, str]] = []

        if code is None:
            for _ in fixture.test_cases:
                test_details.append((False, "Could not extract function from response"))
        else:
            for tc in fixture.test_cases:
                passed, detail = run_test_case(code, fixture.function_name, tc, fixture)
                test_details.append((passed, detail))

        tests_passed = sum(1 for p, _ in test_details if p)
        tests_total = len(fixture.test_cases)
        pass_rate = round(tests_passed / tests_total, 3) if tests_total > 0 else 0.0
        tr.accuracy_score = pass_rate

        fr = CodegenFixtureResult(
            fixture=fixture, task_result=tr,
            tests_passed=tests_passed, tests_total=tests_total,
            pass_rate=pass_rate, extracted_code=code, test_details=test_details,
        )
        results.append(fr)
        if on_fixture_done:
            on_fixture_done(fr)

    return CodegenRunSummary(
        run_id=effective_run_id,
        provider=provider.name,
        model=model or getattr(provider, "default_model", None) or "",
        started_at=started,
        completed_at=datetime.now(UTC),
        fixture_results=results,
    )
