"""Tests for the model class taxonomy."""

from atomics.model_classes import (
    ModelClass,
    classify_model,
    get_models_for_class,
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


def test_model_class_enum_values():
    assert ModelClass.LIGHT == "light"
    assert ModelClass.MID == "mid"
    assert ModelClass.HEAVY == "heavy"
    assert ModelClass.UNKNOWN == "unknown"
