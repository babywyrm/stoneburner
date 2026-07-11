"""Adversarial LLM resilience evaluation.

`ALL_FIXTURES` is the single source of truth for the full fixture set. The
runner and CLI both select from it via `select_fixtures`, so fixture counts are
consistent everywhere (header, progress bar, actual run, and docs).
"""

from __future__ import annotations

from collections.abc import Iterable

from .agentic_reasoning import AGENTIC_REASONING_FIXTURES  # noqa: F401
from .encoding_obfuscation import ENCODING_OBFUSCATION_FIXTURES  # noqa: F401
from .fixtures import ADVERSARIAL_FIXTURES, AdversarialFixture  # noqa: F401
from .many_shot import MANYSHOT_FIXTURES  # noqa: F401
from .mcp_agentic import MCP_AGENTIC_FIXTURES, TOOL_SAFETY_FIXTURES  # noqa: F401
from .multiturn import MULTITURN_FIXTURES  # noqa: F401
from .rag_poisoning import RAG_POISONING_FIXTURES  # noqa: F401
from .tool_desc_injection import TOOL_DESC_INJECTION_FIXTURES  # noqa: F401
from .zerotrust import ZEROTRUST_FIXTURES  # noqa: F401

# Order matters — it is the stable iteration/reporting order.
ALL_FIXTURES: list[AdversarialFixture] = (
    ADVERSARIAL_FIXTURES
    + ZEROTRUST_FIXTURES
    + AGENTIC_REASONING_FIXTURES
    + MCP_AGENTIC_FIXTURES
    + TOOL_SAFETY_FIXTURES
    + MULTITURN_FIXTURES
    + RAG_POISONING_FIXTURES
    + TOOL_DESC_INJECTION_FIXTURES
    + ENCODING_OBFUSCATION_FIXTURES
    + MANYSHOT_FIXTURES
)

# Group aliases expand to the set of categories in each suite, so `--category
# zerotrust` runs every zerotrust fixture without enumerating each category.
GROUP_ALIASES: dict[str, set[str]] = {
    "zerotrust": {f.category for f in ZEROTRUST_FIXTURES},
    "agentic": {f.category for f in AGENTIC_REASONING_FIXTURES},
    "mcp": {f.category for f in MCP_AGENTIC_FIXTURES},
    "tool_safety": {f.category for f in TOOL_SAFETY_FIXTURES},
    "multiturn": {f.category for f in MULTITURN_FIXTURES},
    "rag_poisoning": {f.category for f in RAG_POISONING_FIXTURES},
    "tool_desc_injection": {f.category for f in TOOL_DESC_INJECTION_FIXTURES},
    "encoding": {f.category for f in ENCODING_OBFUSCATION_FIXTURES},
    "many_shot": {f.category for f in MANYSHOT_FIXTURES},
}


def expand_categories(categories: Iterable[str]) -> set[str]:
    """Expand group aliases into their constituent concrete categories."""
    expanded: set[str] = set()
    for c in categories:
        expanded.update(GROUP_ALIASES.get(c, {c}))
    return expanded


def select_fixtures(
    categories: Iterable[str] | None = None,
) -> list[AdversarialFixture]:
    """Return fixtures filtered by category (group aliases expanded).

    With no categories, returns the full `ALL_FIXTURES` set in order.
    """
    if not categories:
        return list(ALL_FIXTURES)
    wanted = expand_categories(categories)
    return [f for f in ALL_FIXTURES if f.category in wanted]


__all__ = [
    "AdversarialFixture",
    "ADVERSARIAL_FIXTURES",
    "ZEROTRUST_FIXTURES",
    "AGENTIC_REASONING_FIXTURES",
    "MCP_AGENTIC_FIXTURES",
    "TOOL_SAFETY_FIXTURES",
    "MULTITURN_FIXTURES",
    "RAG_POISONING_FIXTURES",
    "TOOL_DESC_INJECTION_FIXTURES",
    "ENCODING_OBFUSCATION_FIXTURES",
    "MANYSHOT_FIXTURES",
    "ALL_FIXTURES",
    "GROUP_ALIASES",
    "expand_categories",
    "select_fixtures",
]
