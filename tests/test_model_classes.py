"""Tests for the model class taxonomy."""

from atomics.model_classes import (
    ModelClass,
    classify_model,
    get_models_for_class,
    supports_thinking,
)


def test_classify_known_light_models():
    assert classify_model("claude-haiku-4-5-20251001") == ModelClass.LIGHT
    assert classify_model("gpt-4o-mini") == ModelClass.LIGHT
    assert classify_model("gpt-4.1-nano") == ModelClass.LIGHT
    assert classify_model("codex-mini-latest") == ModelClass.LIGHT


def test_classify_known_mid_models():
    assert classify_model("claude-sonnet-4-6") == ModelClass.MID
    assert classify_model("gpt-4o") == ModelClass.MID
    assert classify_model("us.anthropic.claude-sonnet-4-6") == ModelClass.MID
    assert classify_model("o4-mini") == ModelClass.MID


def test_classify_known_heavy_models():
    assert classify_model("claude-opus-4-6") == ModelClass.HEAVY
    assert classify_model("o3") == ModelClass.HEAVY
    assert classify_model("us.anthropic.claude-opus-4-6-v1") == ModelClass.HEAVY


def test_classify_unknown_model():
    assert classify_model("some-future-model-v99") == ModelClass.UNKNOWN


def test_classify_unmapped_size_tag_is_light_below_two_billion():
    assert classify_model("brand-new:0.8b") == ModelClass.LIGHT
    assert classify_model("brand-new:1.5b") == ModelClass.LIGHT


def test_classify_unmapped_size_tag_is_mid_through_fourteen_billion():
    assert classify_model("brand-new:2b") == ModelClass.MID
    assert classify_model("brand-new:e4b") == ModelClass.MID
    assert classify_model("brand-new:14b") == ModelClass.MID


def test_classify_unmapped_size_tag_is_heavy_above_fourteen_billion():
    assert classify_model("brand-new:24b") == ModelClass.HEAVY
    assert classify_model("brand-new:35b-a3b") == ModelClass.HEAVY


def test_classify_exact_map_wins_over_size_heuristic():
    assert classify_model("custom-agent:latest") == ModelClass.LIGHT


def test_get_models_for_class_light():
    models = get_models_for_class(ModelClass.LIGHT)
    assert "gpt-4o-mini" in models
    assert "claude-haiku-4-5-20251001" in models
    assert "gpt-4o" not in models


def test_get_models_for_class_heavy():
    models = get_models_for_class(ModelClass.HEAVY)
    assert "o3" in models
    assert "claude-opus-4-6" in models
    assert len(models) >= 3


def test_get_models_for_unknown_is_empty():
    assert get_models_for_class(ModelClass.UNKNOWN) == []


def test_classify_ollama_local_models():
    """All gateway models should be classified, not UNKNOWN."""
    assert classify_model("gemma3:4b") == ModelClass.MID
    assert classify_model("gemma4:e4b") == ModelClass.MID
    assert classify_model("functiongemma:latest") == ModelClass.LIGHT
    assert classify_model("llama3.2:1b") == ModelClass.LIGHT
    assert classify_model("qwen3.5:2b") == ModelClass.MID
    assert classify_model("ministral-3:3b") == ModelClass.MID
    assert classify_model("ministral-3:8b") == ModelClass.MID
    assert classify_model("qwen3:8b") == ModelClass.MID
    assert classify_model("qwen3.5:9b") == ModelClass.MID
    assert classify_model("granite4.1:3b") == ModelClass.MID
    assert classify_model("granite4.1:8b") == ModelClass.MID
    assert classify_model("lfm2.5:8b") == ModelClass.MID
    assert classify_model("mistral-nemo:12b") == ModelClass.MID
    assert classify_model("mistral-small:24b") == ModelClass.HEAVY
    assert classify_model("nemotron-3-nano:4b") == ModelClass.MID
    assert classify_model("phi4-mini-reasoning:3.8b") == ModelClass.MID
    assert classify_model("qwen3.6:27b") == ModelClass.HEAVY
    assert classify_model("qwen3.6:35b-a3b") == ModelClass.HEAVY
    assert classify_model("qwen3.8:27b") == ModelClass.HEAVY
    assert classify_model("smollm2:1.7b") == ModelClass.LIGHT
    assert classify_model("phi4-mini:3.8b") == ModelClass.MID
    assert classify_model("phi4:14b") == ModelClass.MID
    assert classify_model("dolphin3:8b") == ModelClass.MID
    assert classify_model("deepseek-r1:14b") == ModelClass.MID
    assert classify_model("custom-agent:latest") == ModelClass.LIGHT


def test_classify_local_gateway_lineup_fully_tagged():
    """Every model the local gateway serves must classify (never UNKNOWN),
    so compare/sweep tables don't show blanks."""
    gateway_models = [
        "cogito:3b", "deepseek-r1:7b", "deepseek-r1:14b", "dolphin3:8b", "dolphin3:latest",
        "functiongemma:latest", "gemma3:4b", "gemma4:12b", "gemma4:26b",
        "gemma4:e4b", "custom-agent:latest", "granite4.1:3b", "granite4.1:8b",
        "llama3.2:1b", "ministral-3:3b", "ministral-3:8b",
        "mistral:7b", "mistral-small3.2:24b", "phi4-mini:3.8b", "phi4-mini:latest",
        "phi4:latest", "phi4-reasoning:14b",
        "qwen2.5-coder:14b", "qwen2.5:1.5b", "qwen2.5:14b", "qwen2.5:3b",
        "qwen2.5:7b", "qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b",
        "qwen3:1.7b", "qwen3:8b", "qwen3:14b", "qwen3:4b",
        "lfm2.5:8b", "mistral-nemo:12b", "mistral-small:24b",
        "nemotron-3-nano:4b", "phi4-mini-reasoning:3.8b",
        "qwen3.6:27b", "qwen3.6:35b-a3b", "qwen3.8:27b", "smollm2:1.7b",
    ]
    unknown = [m for m in gateway_models if classify_model(m) == ModelClass.UNKNOWN]
    assert unknown == [], f"unclassified gateway models: {unknown}"


def test_thinking_support_qwen3_5_family():
    """qwen3.5 models should be thinking-capable."""
    assert supports_thinking("qwen3.5:0.8b") is True
    assert supports_thinking("qwen3.5:2b") is True
    assert supports_thinking("qwen3.5:9b") is True
    assert supports_thinking("qwen3:8b") is True
    assert supports_thinking("qwen3.6:27b") is True
    assert supports_thinking("qwen3.8:27b") is True
    assert supports_thinking("phi4-mini-reasoning:3.8b") is True


def test_thinking_support_deepseek_r1_all_sizes():
    """All deepseek-r1 sizes should be thinking-capable."""
    assert supports_thinking("deepseek-r1:14b") is True
    assert supports_thinking("deepseek-r1:32b") is True
    assert supports_thinking("deepseek-r1:70b") is True


def test_thinking_support_phi4_not_thinking():
    """phi4 models don't use <think> tags."""
    assert supports_thinking("phi4-mini:3.8b") is False
    assert supports_thinking("phi4:14b") is False


def test_thinking_support_gemma_not_thinking():
    """gemma models don't use <think> tags."""
    assert supports_thinking("gemma3:4b") is False
    assert supports_thinking("gemma4:e4b") is False


def test_model_class_enum_values():
    assert ModelClass.LIGHT == "light"
    assert ModelClass.MID == "mid"
    assert ModelClass.HEAVY == "heavy"
    assert ModelClass.UNKNOWN == "unknown"
