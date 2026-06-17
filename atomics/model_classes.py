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
    "claude-opus-4-5": ModelClass.HEAVY,
    "claude-opus-4-6": ModelClass.HEAVY,
    "claude-opus-4-7": ModelClass.HEAVY,
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
    "gpt-5": ModelClass.HEAVY,
    "gpt-5-turbo": ModelClass.MID,
    "gpt-5.3": ModelClass.HEAVY,
    "gpt-5.5": ModelClass.HEAVY,
    "gpt-5.5-turbo": ModelClass.MID,
    "o3": ModelClass.HEAVY,
    "o3-mini": ModelClass.MID,
    "o3-pro": ModelClass.HEAVY,
    "o4-mini": ModelClass.MID,
    "codex-mini-latest": ModelClass.LIGHT,
    # Ollama / local models
    "qwen2.5:1.5b": ModelClass.LIGHT,
    "qwen3:0.6b": ModelClass.LIGHT,
    "qwen3:1.7b": ModelClass.LIGHT,
    "qwen3.5:0.8b": ModelClass.LIGHT,
    "qwen3.5:2b": ModelClass.MID,
    "qwen2.5:3b": ModelClass.MID,
    "qwen3:4b": ModelClass.MID,
    "qwen2.5:7b": ModelClass.MID,
    "qwen2.5:14b": ModelClass.MID,
    "qwen2.5:32b": ModelClass.HEAVY,
    "qwen2.5:72b": ModelClass.HEAVY,
    "llama3.2:1b": ModelClass.LIGHT,
    "llama3.2:3b": ModelClass.MID,
    "llama3.1:8b": ModelClass.MID,
    "mistral:7b": ModelClass.MID,
    "codellama:7b": ModelClass.MID,
    "gemma3:4b": ModelClass.MID,
    "gemma4:e4b": ModelClass.MID,
    "gemma4:12b": ModelClass.MID,
    "gemma4:26b": ModelClass.HEAVY,
    "functiongemma:latest": ModelClass.LIGHT,
    "custom-agent:latest": ModelClass.LIGHT,
    "cogito:3b": ModelClass.MID,
    "ministral-3:3b": ModelClass.MID,
    "deepseek-r1:14b": ModelClass.MID,
    "deepseek-r1:32b": ModelClass.HEAVY,
    "deepseek-r1:70b": ModelClass.HEAVY,
    "phi4-mini:3.8b": ModelClass.MID,
    "phi4-mini:latest": ModelClass.MID,
    "phi4:14b": ModelClass.MID,
    "phi4:latest": ModelClass.MID,
    "qwen2.5-coder:14b": ModelClass.MID,
    "qwen3:14b": ModelClass.MID,
    "qwen3.5:4b": ModelClass.MID,
    "dolphin3:8b": ModelClass.MID,
    "dolphin3:latest": ModelClass.MID,
    "deepseek-r1:7b": ModelClass.MID,
    "phi4-reasoning:14b": ModelClass.MID,
    "mistral-small3.2:24b": ModelClass.HEAVY,
}


THINKING_CAPABLE: frozenset[str] = frozenset({
    # Claude — extended thinking via API
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-sonnet-4-20250514",
    "claude-opus-4-5",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-5",
    # Claude via Bedrock
    "us.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "us.anthropic.claude-opus-4-6-v1",
    # OpenAI reasoning models
    "o3", "o3-mini", "o3-pro", "o4-mini",
    "gpt-5", "gpt-5-turbo", "gpt-5.3", "gpt-5.5",
    # Ollama — qwen3 family and deepseek-r1 use <think> tags
    "qwen3:0.6b", "qwen3:1.7b", "qwen3:4b", "qwen3:14b", "qwen3:32b", "qwen3:72b",
    "qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b",
    "deepseek-r1:7b", "deepseek-r1:14b", "deepseek-r1:32b", "deepseek-r1:70b",
    "phi4-reasoning:14b",
})

# Default thinking budget (tokens) per provider family
THINKING_BUDGET_DEFAULTS: dict[str, int] = {
    "claude": 10_000,
    "openai": 8_192,
    "ollama": 4_096,
}


def supports_thinking(model_id: str) -> bool:
    """Check if a model supports thinking/reasoning mode."""
    if model_id in THINKING_CAPABLE:
        return True
    for prefix in ("qwen3", "o3", "o4", "deepseek-r1"):
        if model_id.startswith(prefix):
            return True
    return False


def classify_model(model_id: str) -> ModelClass:
    """Return the class for a model ID, or UNKNOWN if not in the registry."""
    return MODEL_CLASS_MAP.get(model_id, ModelClass.UNKNOWN)


def get_models_for_class(cls: ModelClass) -> list[str]:
    """Return all known model IDs in a given class."""
    return [m for m, c in MODEL_CLASS_MAP.items() if c == cls]
