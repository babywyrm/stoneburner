"""Shared reasoning-effort vocabulary, mapped to each provider's native fields.

The CLI/API speak one dialect (``--effort``, ``--reasoning-mode``). Each
adapter turns that into the payload its API actually accepts, and records
that payload on ``ProviderResponse.reasoning_request`` so a sweep can say
what was sent, not just what the operator typed.
"""

from __future__ import annotations

from typing import Literal

EffortLevel = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
ReasoningMode = Literal["standard", "pro"]

CANONICAL_EFFORTS: tuple[str, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
EFFORT_ALIASES: dict[str, str] = {
    "xl": "xhigh",
    "x-high": "xhigh",
    "x_high": "xhigh",
    "ultra": "max",
    "min": "minimal",
}
CANONICAL_MODES: tuple[str, ...] = ("standard", "pro")

# Claude 4.6 lists low/medium/high/max. Newer families advertise xhigh.
_CLAUDE_XHIGH_PREFIXES: tuple[str, ...] = (
    "claude-fable-5",
    "claude-mythos",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
)


class EffortError(ValueError):
    """Operator-facing value that is not a known effort or reasoning mode."""


def normalize_effort(value: str | None) -> str | None:
    """Canonicalize an operator effort string, or None when unset."""
    if value is None:
        return None
    key = value.strip().lower()
    if not key:
        return None
    key = EFFORT_ALIASES.get(key, key)
    if key not in CANONICAL_EFFORTS:
        allowed = ", ".join((*CANONICAL_EFFORTS, "xl", "ultra"))
        raise EffortError(f"unknown effort {value!r}; expected {allowed}")
    return key


def normalize_reasoning_mode(value: str | None) -> str | None:
    """Canonicalize OpenAI ``reasoning.mode``, or None when unset."""
    if value is None:
        return None
    key = value.strip().lower()
    if not key:
        return None
    if key not in CANONICAL_MODES:
        raise EffortError(f"unknown reasoning mode {value!r}; expected standard or pro")
    return key


def openai_reasoning(
    effort: str | None,
    mode: str | None,
    *,
    thinking: bool | None = None,
) -> dict[str, str] | None:
    """Responses API ``reasoning`` object, or None when nothing to send."""
    resolved_effort = normalize_effort(effort)
    resolved_mode = normalize_reasoning_mode(mode)
    if resolved_effort is None and thinking is False:
        resolved_effort = "none"
    if resolved_effort is None and resolved_mode is None:
        return None
    payload: dict[str, str] = {}
    if resolved_effort is not None:
        payload["effort"] = resolved_effort
    if resolved_mode is not None:
        payload["mode"] = resolved_mode
    return payload


def openai_chat_effort(effort: str | None) -> str | None:
    """Chat Completions ``reasoning_effort`` scalar."""
    payload = openai_reasoning(effort, None)
    if payload is None:
        return None
    return payload.get("effort")


def apply_chat_effort(body: dict, effort: str | None) -> dict[str, str] | None:
    """Set ``reasoning_effort`` on an OpenAI-compatible chat body."""
    value = openai_chat_effort(effort)
    if value is None:
        return None
    body["reasoning_effort"] = value
    return {"effort": value}


def claude_effort_value(effort: str | None, model: str) -> str | None:
    """Anthropic ``output_config.effort`` for this model, or None."""
    resolved = normalize_effort(effort)
    if resolved is None or resolved == "none":
        return None
    if resolved == "minimal":
        return "low"
    if resolved == "xhigh" and not _claude_has_xhigh(model):
        return "max"
    return resolved


def claude_request(
    *,
    model: str,
    thinking: bool | None,
    thinking_budget: int | None,
    effort: str | None,
    default_budget: int = 10_000,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    """Native Claude thinking block and extra request kwargs.

    Effort on a 4.6+ model uses adaptive thinking. ``--thinking`` without
    effort keeps the older ``budget_tokens`` form so existing commands do
    not change payload. ``--no-thinking`` suppresses the thinking block
    but still forwards ``output_config.effort`` when the operator set one.
    """
    extra: dict[str, object] = {}
    native_effort = claude_effort_value(effort, model)
    if native_effort is not None:
        extra["output_config"] = {"effort": native_effort}

    if thinking is False:
        return None, extra

    if native_effort is not None:
        return {"type": "adaptive"}, extra

    if thinking is True:
        budget = thinking_budget or default_budget
        return {"type": "enabled", "budget_tokens": budget}, extra

    return None, extra


def _claude_has_xhigh(model: str) -> bool:
    return any(model.startswith(prefix) for prefix in _CLAUDE_XHIGH_PREFIXES)
