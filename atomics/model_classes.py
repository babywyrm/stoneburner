"""Model class taxonomy for fair cross-provider comparison.

Maps model IDs to a class (light/mid/heavy) so the compare command can
flag mismatched comparisons and users can filter by class.
"""

from __future__ import annotations

from enum import StrEnum


class ModelClass(StrEnum):
    LIGHT = "light"
    MID = "mid"
    HEAVY = "heavy"
    UNKNOWN = "unknown"


MODEL_CLASS_MAP: dict[str, ModelClass] = {
    # Claude via Anthropic API
    "claude-haiku-4-5-20251001": ModelClass.LIGHT,
    "claude-sonnet-4-6": ModelClass.MID,
    "claude-sonnet-4-20250514": ModelClass.MID,
    "claude-opus-4-6": ModelClass.HEAVY,
    # Claude via Bedrock (inference profiles)
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": ModelClass.LIGHT,
    "us.anthropic.claude-sonnet-4-6": ModelClass.MID,
    "us.anthropic.claude-sonnet-4-20250514-v1:0": ModelClass.MID,
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": ModelClass.MID,
    "us.anthropic.claude-opus-4-6-v1": ModelClass.HEAVY,
    "anthropic.claude-3-5-sonnet-20241022-v2:0": ModelClass.MID,
    "anthropic.claude-3-5-haiku-20241022-v1:0": ModelClass.LIGHT,
    "anthropic.claude-sonnet-4-20250514-v1:0": ModelClass.MID,
    # OpenAI
    "gpt-4o": ModelClass.MID,
    "gpt-4o-2024-11-20": ModelClass.MID,
    "gpt-4o-mini": ModelClass.LIGHT,
    "gpt-4o-mini-2024-07-18": ModelClass.LIGHT,
    "gpt-4.1": ModelClass.MID,
    "gpt-4.1-mini": ModelClass.LIGHT,
    "gpt-4.1-nano": ModelClass.LIGHT,
    "o3": ModelClass.HEAVY,
    "o3-mini": ModelClass.MID,
    "o4-mini": ModelClass.MID,
    "codex-mini-latest": ModelClass.LIGHT,
}


def classify_model(model_id: str) -> ModelClass:
    """Return the class for a model ID, or UNKNOWN if not in the registry."""
    return MODEL_CLASS_MAP.get(model_id, ModelClass.UNKNOWN)


def get_models_for_class(cls: ModelClass) -> list[str]:
    """Return all known model IDs in a given class."""
    return [m for m, c in MODEL_CLASS_MAP.items() if c == cls]
