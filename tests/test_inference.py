"""Tests for the inference.env standard reader/resolver (atomics.inference)."""

from __future__ import annotations

import textwrap

import pytest

from atomics import inference

# ── parse_env ─────────────────────────────────────────────────────────────────

def test_parse_env_ignores_comments_and_blanks():
    text = "# a comment\n\nINFERENCE_MODEL=gemma3:4b\nINFERENCE_THINK=false\n"
    assert inference.parse_env(text) == {
        "INFERENCE_MODEL": "gemma3:4b",
        "INFERENCE_THINK": "false",
    }


def test_parse_env_keeps_value_with_equals_and_colon():
    text = "INFERENCE_URL=http://h:8000/v1\nINFERENCE_API_KEY=a=b=c\n"
    parsed = inference.parse_env(text)
    assert parsed["INFERENCE_URL"] == "http://h:8000/v1"
    assert parsed["INFERENCE_API_KEY"] == "a=b=c"


# ── normalize_legacy ──────────────────────────────────────────────────────────

def test_normalize_canonical_passthrough():
    raw = {"INFERENCE_BACKEND": "ollama", "INFERENCE_URL": "http://h:11434",
           "INFERENCE_MODEL": "m", "INFERENCE_THINK": "true"}
    norm = inference.normalize_legacy(raw)
    assert norm["INFERENCE_BACKEND"] == "ollama"
    assert norm["INFERENCE_URL"] == "http://h:11434"
    assert norm["INFERENCE_THINK"] == "true"


def test_normalize_legacy_ollama_keys():
    raw = {"INFERENCE_API": "ollama", "OLLAMA_URL": "http://h:11434",
           "OLLAMA_MODEL": "qwen2.5:3b", "OLLAMA_THINK": "false"}
    norm = inference.normalize_legacy(raw)
    assert norm["INFERENCE_BACKEND"] == "ollama"
    assert norm["INFERENCE_URL"] == "http://h:11434"
    assert norm["INFERENCE_MODEL"] == "qwen2.5:3b"
    assert norm["INFERENCE_THINK"] == "false"


def test_normalize_legacy_openai_maps_to_vllm_local_gateway():
    raw = {"INFERENCE_API": "openai", "OPENAI_BASE_URL": "http://gpu:8000/v1",
           "OPENAI_MODEL": "qwen2.5:3b", "OPENAI_API_KEY": "dummy"}
    norm = inference.normalize_legacy(raw)
    assert norm["INFERENCE_BACKEND"] == "vllm"
    assert norm["INFERENCE_URL"] == "http://gpu:8000/v1"
    assert norm["INFERENCE_MODEL"] == "qwen2.5:3b"
    assert norm["INFERENCE_API_KEY"] == "dummy"


def test_normalize_canonical_wins_over_legacy():
    raw = {"INFERENCE_BACKEND": "ollama", "INFERENCE_MODEL": "canonical",
           "INFERENCE_API": "openai", "OPENAI_MODEL": "legacy"}
    norm = inference.normalize_legacy(raw)
    assert norm["INFERENCE_BACKEND"] == "ollama"
    assert norm["INFERENCE_MODEL"] == "canonical"


# ── InferenceTarget / load_control_file ───────────────────────────────────────

def test_target_from_text_full():
    text = textwrap.dedent("""\
        INFERENCE_DIFFICULTY=easy
        INFERENCE_POOL=brainbox
        INFERENCE_BACKEND=ollama
        INFERENCE_URL=http://10.0.0.9:11434
        INFERENCE_MODEL=gemma3:4b
        INFERENCE_THINK=false
        INFERENCE_API_KEY=
        INFERENCE_RESOLVED_BY=control-plane-resolver
    """)
    t = inference.InferenceTarget.from_text(text)
    assert t.backend == "ollama"
    assert t.url == "http://10.0.0.9:11434"
    assert t.model == "gemma3:4b"
    assert t.think is False
    assert t.api_key == ""
    assert t.difficulty == "easy"
    assert t.pool == "brainbox"
    assert t.resolved_by == "control-plane-resolver"


def test_target_think_parsing():
    assert inference.InferenceTarget.from_text("INFERENCE_THINK=true").think is True
    assert inference.InferenceTarget.from_text("INFERENCE_THINK=false").think is False
    assert inference.InferenceTarget.from_text("INFERENCE_MODEL=x").think is False


def test_load_control_file_explicit_path(tmp_path):
    p = tmp_path / "inference.env"
    p.write_text("INFERENCE_BACKEND=vllm\nINFERENCE_URL=http://g:8000/v1\nINFERENCE_MODEL=m\n")
    t = inference.load_control_file(str(p))
    assert t is not None
    assert t.backend == "vllm"
    assert t.url == "http://g:8000/v1"


def test_load_control_file_missing_returns_none(tmp_path):
    assert inference.load_control_file(str(tmp_path / "nope.env")) is None


def test_load_control_file_searches_env_override(tmp_path, monkeypatch):
    p = tmp_path / "inference.env"
    p.write_text("INFERENCE_BACKEND=ollama\nINFERENCE_MODEL=m\nINFERENCE_URL=http://h:11434\n")
    monkeypatch.setenv("INFERENCE_ENV", str(p))
    t = inference.load_control_file()
    assert t is not None and t.backend == "ollama"


def test_load_control_file_none_when_no_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("INFERENCE_ENV", raising=False)
    monkeypatch.delenv("BRAIN_ENV", raising=False)
    # point the default search paths at nonexistent files
    monkeypatch.setattr(inference, "_DEFAULT_PATHS",
                        (str(tmp_path / "a"), str(tmp_path / "b")))
    assert inference.load_control_file() is None


def test_load_control_file_normalizes_legacy(tmp_path):
    p = tmp_path / "legacy.env"
    p.write_text("INFERENCE_API=openai\nOPENAI_BASE_URL=http://g:8000/v1\nOPENAI_MODEL=m\n")
    t = inference.load_control_file(str(p))
    assert t is not None
    assert t.backend == "vllm"
    assert t.url == "http://g:8000/v1"


# ── resolver (agnostic) ───────────────────────────────────────────────────────

def test_resolve_model_picks_tier():
    machine = {"difficulty_models": {"easy": "gemma3:4b", "hard": "qwen2.5:1.5b"}}
    assert inference.resolve_model(machine, "easy") == "gemma3:4b"


def test_resolve_model_unknown_tier_raises():
    with pytest.raises(ValueError):
        inference.resolve_model({"difficulty_models": {"easy": "x"}}, "medium")


def test_resolve_endpoint_defaults_backend_ollama():
    ep = inference.resolve_endpoint({"endpoint": {"host": "h", "port": 8000}})
    assert ep["url"] == "http://h:8000"
    assert ep["backend"] == "ollama"


def test_check_backend():
    assert inference.check_backend({"supported_backends": ["ollama"]}, "ollama") is True
    assert inference.check_backend({"supported_backends": ["ollama"]}, "vllm") is False
    assert inference.check_backend({}, "ollama") is True


def test_check_model_compat_grouped():
    machine = {"model_compatibility": {"fully_solvable": ["a"], "incompatible_x": ["b"]}}
    assert inference.check_model_compat(machine, "a")[0] == "OK"
    assert inference.check_model_compat(machine, "b")[0] == "INCOMPATIBLE"
    assert inference.check_model_compat(machine, "c")[0] == "UNTESTED"


def test_resolve_and_render_roundtrip():
    machine = {"difficulty_models": {"easy": "gemma3:4b"}, "think_default": False,
               "supported_backends": ["ollama"],
               "model_compatibility": {"compatible": ["gemma3:4b"]}}
    profile = {"endpoint": {"host": "h", "port": 11434, "url": "http://h:11434"},
               "backend": "ollama"}
    out = inference.resolve(machine, profile, "easy", "brainbox", "stoneburner")
    assert out["resolved"]["model"] == "gemma3:4b"
    assert out["backend_ok"] is True
    assert out["compat"][0] == "OK"
    # the rendered env must round-trip back into an equivalent target
    t = inference.InferenceTarget.from_text(out["env"])
    assert t.backend == "ollama"
    assert t.model == "gemma3:4b"
    assert t.difficulty == "easy"
    assert t.pool == "brainbox"


# ── provider_from_target (auto-load integration point) ────────────────────────

def test_provider_from_target_ollama():
    t = inference.InferenceTarget(backend="ollama", url="http://h:11434", model="m")
    p = inference.provider_from_target(t, client=object())
    assert p.name == "ollama"


def test_provider_from_target_vllm():
    t = inference.InferenceTarget(backend="vllm", url="http://g:8000/v1",
                                  model="m", api_key="dummy")
    p = inference.provider_from_target(t, client=object())
    assert p.name == "vllm"


def test_provider_from_target_openai():
    t = inference.InferenceTarget(backend="openai", url="", model="gpt-4o",
                                  api_key="sk-x")
    p = inference.provider_from_target(t, client=object())
    assert p.name == "openai"


def test_provider_from_target_unknown_backend_raises():
    t = inference.InferenceTarget(backend="nope", url="", model="m")
    with pytest.raises(ValueError):
        inference.provider_from_target(t, client=object())
