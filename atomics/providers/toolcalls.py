"""Provider-layer representation of a structured tool call.

This lives in the providers layer rather than beside the toolcall eval suite
because `ProviderResponse` references it, and `tests/test_layering.py` forbids
the providers package from importing `atomics.eval`. Putting it in the eval
module would repeat the layering mistake the 0.13.0 outcomes split corrected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def parse_arguments(raw: Any) -> tuple[dict[str, Any], bool]:
    """Return `(arguments, malformed)` from a dialect's raw argument payload.

    OpenAI and Ollama deliver arguments as a JSON *string*; Anthropic delivers an
    already-parsed object. Malformed input yields empty arguments and a flag
    rather than an exception: a model emitting unparseable calls is a result
    worth recording, not an error to abort a run over.
    """
    if isinstance(raw, dict):
        return dict(raw), False
    if raw is None or raw == "":
        return {}, False
    if not isinstance(raw, str):
        return {}, True
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}, True
    if not isinstance(parsed, dict):
        return {}, True
    return parsed, False


@dataclass(frozen=True)
class ToolCall:
    """One structured call a model asked to make. Never executed."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    malformed: bool = False
    raw: dict | None = field(default=None, repr=False)
