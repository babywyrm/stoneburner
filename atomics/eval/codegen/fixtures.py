"""Code generation fixtures — 15 Python programming tasks with test cases.

Each fixture describes a function to implement and provides deterministic
test cases for automated correctness checking. No judge needed.

Complexity spread: 5 LIGHT (basic algorithms), 5 MODERATE (data structures,
string processing), 5 HEAVY (multi-step logic, edge cases).
"""

from __future__ import annotations

from atomics.eval.codegen import CodegenFixture, CodeTestCase
from atomics.models import TaskComplexity

CODEGEN_FIXTURES: list[CodegenFixture] = [
    # ── LIGHT ─────────────────────────────────────────────────────────────────
    CodegenFixture(
        id="cg-01",
        complexity=TaskComplexity.LIGHT,
        language="python",
        function_name="fizzbuzz",
        description="Return 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for both, otherwise the number as a string.",
        signature="def fizzbuzz(n: int) -> str:",
        test_cases=[
            CodeTestCase([1], "1"),
            CodeTestCase([3], "Fizz"),
            CodeTestCase([5], "Buzz"),
            CodeTestCase([15], "FizzBuzz"),
            CodeTestCase([7], "7"),
            CodeTestCase([30], "FizzBuzz"),
        ],
    ),
    CodegenFixture(
        id="cg-02",
        complexity=TaskComplexity.LIGHT,
        language="python",
        function_name="is_palindrome",
        description="Check if a string is a palindrome (case-insensitive, ignoring spaces).",
        signature="def is_palindrome(s: str) -> bool:",
        test_cases=[
            CodeTestCase(["racecar"], True),
            CodeTestCase(["hello"], False),
            CodeTestCase(["A man a plan a canal Panama"], True),
            CodeTestCase([""], True),
            CodeTestCase(["Madam"], True),
        ],
    ),
    CodegenFixture(
        id="cg-03",
        complexity=TaskComplexity.LIGHT,
        language="python",
        function_name="fibonacci",
        description="Return the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1).",
        signature="def fibonacci(n: int) -> int:",
        test_cases=[
            CodeTestCase([0], 0),
            CodeTestCase([1], 1),
            CodeTestCase([5], 5),
            CodeTestCase([10], 55),
            CodeTestCase([20], 6765),
        ],
    ),
    CodegenFixture(
        id="cg-04",
        complexity=TaskComplexity.LIGHT,
        language="python",
        function_name="reverse_words",
        description="Reverse the order of words in a string. Multiple spaces should become single spaces.",
        signature="def reverse_words(s: str) -> str:",
        test_cases=[
            CodeTestCase(["hello world"], "world hello"),
            CodeTestCase(["  the sky is blue  "], "blue is sky the"),
            CodeTestCase(["a"], "a"),
            CodeTestCase([""], ""),
        ],
    ),
    CodegenFixture(
        id="cg-05",
        complexity=TaskComplexity.LIGHT,
        language="python",
        function_name="count_vowels",
        description="Count the number of vowels (a, e, i, o, u) in a string, case-insensitive.",
        signature="def count_vowels(s: str) -> int:",
        test_cases=[
            CodeTestCase(["hello"], 2),
            CodeTestCase(["AEIOU"], 5),
            CodeTestCase(["rhythm"], 0),
            CodeTestCase([""], 0),
            CodeTestCase(["Python Programming"], 4),
        ],
    ),
    # ── MODERATE ──────────────────────────────────────────────────────────────
    CodegenFixture(
        id="cg-06",
        complexity=TaskComplexity.MODERATE,
        language="python",
        function_name="two_sum",
        description="Given a list of integers and a target, return indices of two numbers that add up to the target. Return as a sorted list of two indices.",
        signature="def two_sum(nums: list[int], target: int) -> list[int]:",
        test_cases=[
            CodeTestCase([[2, 7, 11, 15], 9], [0, 1]),
            CodeTestCase([[3, 2, 4], 6], [1, 2]),
            CodeTestCase([[1, 5, 3, 7], 8], [1, 2]),
        ],
    ),
    CodegenFixture(
        id="cg-07",
        complexity=TaskComplexity.MODERATE,
        language="python",
        function_name="flatten_nested",
        description="Flatten a nested list of integers to a single flat list. Handle arbitrary nesting depth.",
        signature="def flatten_nested(lst: list) -> list[int]:",
        test_cases=[
            CodeTestCase([[[1, [2, 3], [4, [5, 6]]]]], [1, 2, 3, 4, 5, 6]),
            CodeTestCase([[[1, 2, 3]]], [1, 2, 3]),
            CodeTestCase([[[]]], []),
            CodeTestCase([[[1, [2, [3, [4]]]]]], [1, 2, 3, 4]),
        ],
    ),
    CodegenFixture(
        id="cg-08",
        complexity=TaskComplexity.MODERATE,
        language="python",
        function_name="group_anagrams",
        description="Group a list of strings by anagrams. Return a list of groups (order within groups and order of groups doesn't matter).",
        signature="def group_anagrams(strs: list[str]) -> list[list[str]]:",
        test_cases=[
            CodeTestCase(
                [["eat", "tea", "tan", "ate", "nat", "bat"]],
                [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]],
                description="compare as sorted sets",
            ),
            CodeTestCase([[""]], [[""]]),
            CodeTestCase([["a"]], [["a"]]),
        ],
    ),
    CodegenFixture(
        id="cg-09",
        complexity=TaskComplexity.MODERATE,
        language="python",
        function_name="valid_brackets",
        description="Check if a string of brackets ()[]{}is validly nested and matched.",
        signature="def valid_brackets(s: str) -> bool:",
        test_cases=[
            CodeTestCase(["()[]{}"], True),
            CodeTestCase(["(]"], False),
            CodeTestCase(["([{}])"], True),
            CodeTestCase([""], True),
            CodeTestCase(["((())"], False),
            CodeTestCase(["{[()]}"], True),
        ],
    ),
    CodegenFixture(
        id="cg-10",
        complexity=TaskComplexity.MODERATE,
        language="python",
        function_name="merge_intervals",
        description="Merge overlapping intervals. Input: list of [start, end] pairs. Return merged intervals sorted by start.",
        signature="def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:",
        test_cases=[
            CodeTestCase([[[1, 3], [2, 6], [8, 10], [15, 18]]], [[1, 6], [8, 10], [15, 18]]),
            CodeTestCase([[[1, 4], [4, 5]]], [[1, 5]]),
            CodeTestCase([[[1, 2]]], [[1, 2]]),
            CodeTestCase([[]], []),
        ],
    ),
    # ── HEAVY ─────────────────────────────────────────────────────────────────
    CodegenFixture(
        id="cg-11",
        complexity=TaskComplexity.HEAVY,
        language="python",
        function_name="lru_cache_ops",
        description=(
            "Implement an LRU cache with get and put operations. "
            "Input: capacity and a list of operations as tuples ('get', key) or ('put', key, value). "
            "Return a list of results for get operations (-1 if not found)."
        ),
        signature="def lru_cache_ops(capacity: int, operations: list[tuple]) -> list[int]:",
        test_cases=[
            CodeTestCase(
                [2, [("put", 1, 1), ("put", 2, 2), ("get", 1), ("put", 3, 3), ("get", 2), ("get", 3)]],
                [1, -1, 3],
            ),
            CodeTestCase(
                [1, [("put", 1, 10), ("get", 1), ("put", 2, 20), ("get", 1), ("get", 2)]],
                [10, -1, 20],
            ),
        ],
        max_output_tokens=1500,
    ),
    CodegenFixture(
        id="cg-12",
        complexity=TaskComplexity.HEAVY,
        language="python",
        function_name="longest_substring_k",
        description="Find the length of the longest substring with at most k distinct characters.",
        signature="def longest_substring_k(s: str, k: int) -> int:",
        test_cases=[
            CodeTestCase(["eceba", 2], 3),
            CodeTestCase(["aa", 1], 2),
            CodeTestCase(["aabbcc", 2], 4),
            CodeTestCase(["", 1], 0),
            CodeTestCase(["abcdef", 6], 6),
        ],
    ),
    CodegenFixture(
        id="cg-13",
        complexity=TaskComplexity.HEAVY,
        language="python",
        function_name="serialize_tree",
        description=(
            "Serialize and deserialize a binary tree represented as nested tuples. "
            "A node is (value, left, right) where left/right can be None. "
            "serialize returns a string, deserialize reconstructs the tuple tree. "
            "Return deserialize(serialize(tree)) — it should equal the input."
        ),
        signature="def serialize_tree(tree: tuple | None) -> tuple | None:",
        test_cases=[
            CodeTestCase([(1, (2, None, None), (3, None, None))], (1, (2, None, None), (3, None, None))),
            CodeTestCase([None], None),
            CodeTestCase([(1, None, None)], (1, None, None)),
            CodeTestCase([(1, (2, (4, None, None), None), (3, None, (5, None, None)))],
                     (1, (2, (4, None, None), None), (3, None, (5, None, None)))),
        ],
        max_output_tokens=1500,
    ),
    CodegenFixture(
        id="cg-14",
        complexity=TaskComplexity.HEAVY,
        language="python",
        function_name="calculator",
        description="Evaluate a mathematical expression string containing +, -, *, / (integer division), parentheses, and non-negative integers. Return the integer result.",
        signature="def calculator(expression: str) -> int:",
        test_cases=[
            CodeTestCase(["3+2*2"], 7),
            CodeTestCase(["(1+(4+5+2)-3)+(6+8)"], 23),
            CodeTestCase(["2*(5+5*2)/3+(6/2+8)"], 21),
            CodeTestCase(["100"], 100),
        ],
        max_output_tokens=1500,
    ),
    CodegenFixture(
        id="cg-15",
        complexity=TaskComplexity.HEAVY,
        language="python",
        function_name="topological_sort",
        description=(
            "Given n nodes (0 to n-1) and a list of directed edges [from, to], "
            "return a valid topological ordering. Return an empty list if a cycle exists."
        ),
        signature="def topological_sort(n: int, edges: list[list[int]]) -> list[int]:",
        test_cases=[
            CodeTestCase([4, [[1, 0], [2, 0], [3, 1], [3, 2]]], [3, 1, 2, 0], description="any valid topo order"),
            CodeTestCase([2, [[1, 0]]], [1, 0]),
            CodeTestCase([2, [[0, 1], [1, 0]]], []),
        ],
    ),
]

ALL_CODEGEN_FIXTURES = CODEGEN_FIXTURES
