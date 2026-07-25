"""Import-direction rules, enforced so they cannot quietly regress.

Two violations were fixed by hand and would have been easy to reintroduce: the
`distributed` and `api` packages reached up into `atomics.commands` for the
provider factory, and `atomics.providers` imported the outcome contract from
`atomics.eval`. Both are invisible at runtime — the code works, it is only the
dependency direction that is wrong — so nothing but a test like this catches
them.

Checked by parsing the AST rather than importing, so function-level imports
count too and no module side effects run.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ATOMICS = Path(__file__).resolve().parent.parent / "atomics"


def _source_files() -> list[Path]:
    return sorted(p for p in ATOMICS.rglob("*.py") if "__pycache__" not in p.parts)


def _imported_modules(path: Path) -> set[str]:
    """Every module named by an import anywhere in the file, nesting included."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # A relative import cannot cross package boundaries here, so only
            # absolute ones are interesting.
            if node.level == 0 and node.module:
                modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def _rel(path: Path) -> str:
    return path.relative_to(ATOMICS.parent).as_posix()


def _violations(
    files: list[Path], forbidden_prefix: str
) -> list[tuple[str, str]]:
    found = []
    for path in files:
        for module in sorted(_imported_modules(path)):
            if module == forbidden_prefix or module.startswith(f"{forbidden_prefix}."):
                found.append((_rel(path), module))
    return found


def test_only_the_cli_layer_imports_the_command_layer() -> None:
    """`api` and `distributed` must build providers without the CLI.

    `atomics.cli` is the CLI entry point and `atomics/commands/` is the layer
    itself, so both are allowed to import it.
    """
    candidates = [
        path
        for path in _source_files()
        if "commands" not in path.relative_to(ATOMICS).parts
        and path.name != "cli.py"
    ]
    assert candidates, "no files to check — the layout must have moved"

    violations = _violations(candidates, "atomics.commands")
    assert not violations, (
        "non-CLI code imported the command layer: "
        + ", ".join(f"{path} -> {module}" for path, module in violations)
    )


def test_providers_do_not_depend_on_the_eval_layer() -> None:
    """Providers sit below eval, so the outcome contract lives with providers."""
    provider_files = [
        path for path in _source_files() if path.parent.name == "providers"
    ]
    assert provider_files, "no provider modules found — the layout must have moved"

    violations = _violations(provider_files, "atomics.eval")
    assert not violations, (
        "provider code imported the eval layer: "
        + ", ".join(f"{path} -> {module}" for path, module in violations)
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "ProviderOutcome",
        "ProviderOutcomeKind",
        "policy_block_reason",
        "provider_outcome_from_exception",
    ],
)
def test_eval_outcomes_still_re_exports_the_provider_contract(
    module_name: str,
) -> None:
    """Moving the contract must not break its long-standing import path."""
    from atomics.eval import outcomes as eval_outcomes
    from atomics.providers import outcomes as provider_outcomes

    assert getattr(eval_outcomes, module_name) is getattr(
        provider_outcomes, module_name
    )
    assert module_name in eval_outcomes.__all__
