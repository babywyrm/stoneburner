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
    """All gpu-host models should be classified, not UNKNOWN."""
    assert classify_model("gemma3:4b") == ModelClass.MID
    assert classify_model("gemma4:e4b") == ModelClass.MID
    assert classify_model("functiongemma:latest") == ModelClass.LIGHT
    assert classify_model("llama3.2:1b") == ModelClass.LIGHT
    assert classify_model("qwen3.5:2b") == ModelClass.MID
    assert classify_model("ministral-3:3b") == ModelClass.MID
    assert classify_model("phi4-mini:3.8b") == ModelClass.MID
    assert classify_model("phi4:14b") == ModelClass.MID
    assert classify_model("dolphin3:8b") == ModelClass.MID
    assert classify_model("deepseek-r1:14b") == ModelClass.MID
    assert classify_model("custom-agent:latest") == ModelClass.LIGHT


def test_thinking_support_qwen3_5_family():
    """qwen3.5 models should be thinking-capable."""
    assert supports_thinking("qwen3.5:0.8b") is True
    assert supports_thinking("qwen3.5:2b") is True


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
