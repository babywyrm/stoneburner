"""Code generation evaluation — functional correctness via test execution.

Unlike quality-judged evals, codegen fixtures are scored by running the
generated code against deterministic test cases. Pass/fail is objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from atomics.models import TaskComplexity


@dataclass(frozen=True)
class CodeTestCase:
    """A single input/output test case for a codegen fixture."""

    input_args: list  # arguments to pass to the function
    expected_output: object  # expected return value
    description: str = ""


@dataclass(frozen=True)
class CodegenFixture:
    """A code generation evaluation fixture."""

    id: str
    complexity: TaskComplexity
    language: str  # "python" only for now
    function_name: str  # name of the function the model should implement
    description: str  # natural language task description
    signature: str  # function signature hint
    test_cases: list[CodeTestCase]
    max_output_tokens: int = 1024


__all__ = ["CodegenFixture", "CodeTestCase"]
