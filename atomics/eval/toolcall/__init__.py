"""Tool-call divergence suite.

Measures whether a model that refuses an attack in conversation also refuses it
when the same attack arrives as a function call. The tool channel is scored
deterministically — a declared dangerous call either was emitted or was not — so
the headline result carries no judge dependence.

No tool call is ever executed. The suite records what the model asked to do.
"""

from atomics.eval.toolcall.catalog import (
    BENIGN_TOOLS,
    CATALOG,
    PROBE_PROMPT,
    PROBE_TOOL,
    schemas_for,
)
from atomics.eval.toolcall.fixtures import (
    ALL_FIXTURES,
    GROUP_ALIASES,
    DangerousCall,
    ToolCallFixture,
    fixtures_for_category,
)
from atomics.eval.toolcall.scorer import ToolChannelResult, ToolOutcome, classify

__all__ = [
    "ALL_FIXTURES",
    "BENIGN_TOOLS",
    "CATALOG",
    "GROUP_ALIASES",
    "PROBE_PROMPT",
    "PROBE_TOOL",
    "DangerousCall",
    "ToolCallFixture",
    "ToolChannelResult",
    "ToolOutcome",
    "classify",
    "fixtures_for_category",
    "schemas_for",
]
