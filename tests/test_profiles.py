"""Tests for atomics profiles — custom target profiles for AI gate testing."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atomics.profiles import (
    ProfileError,
    TargetProfile,
    _extract_latency,
    _extract_text,
    classify_response,
    load_profile,
    render_body,
)


# ── TargetProfile dataclass ───────────────────────────────────────────────────


class TestTargetProfile:
    def test_ollama_valid(self):
        p = TargetProfile(
            name="test", type="ollama", ollama_host="http://localhost:11434"
        )
        assert p.type == "ollama"
        assert p.num_predict == 2048

    def test_http_valid(self):
        p = TargetProfile(
            name="test", type="http", http_url="http://localhost:5000/classify"
        )
        assert p.type == "http"
        assert p.http_method == "POST"

    def test_invalid_type(self):
        with pytest.raises(ProfileError, match="must be 'ollama' or 'http'"):
            TargetProfile(name="bad", type="grpc")

    def test_ollama_missing_host(self):
        with pytest.raises(ProfileError, match="requires ollama.host"):
            TargetProfile(name="bad", type="ollama")

    def test_http_missing_url(self):
        with pytest.raises(ProfileError, match="requires http.url"):
            TargetProfile(name="bad", type="http")

    def test_defaults(self):
        p = TargetProfile(
            name="t", type="ollama", ollama_host="http://h:11434"
        )
        assert p.model == ""
        assert p.prompts == []
        assert p.classify is None
        assert p.temperature is None
        assert p.system_prompt == ""


# ── YAML loading ──────────────────────────────────────────────────────────────


class TestLoadProfile:
    def _write_yaml(self, tmp: Path, content: str) -> str:
        f = tmp / "profile.yaml"
        f.write_text(content)
        return str(f)

    def test_load_ollama_profile(self, tmp_path):
        path = self._write_yaml(tmp_path, """
name: test-gk
type: ollama
ollama:
  host: "http://gpu-host:11434"
  model: qwen2.5:3b
  system_prompt: "You are a gatekeeper."
  temperature: 0.0
  num_predict: 24
prompts:
  - "Allow pod nginx"
  - "Deny pod miner"
classify:
  approved: ["APPROVE", "ALLOW"]
  blocked: ["DENY", "BLOCK"]
""")
        p = load_profile(path)
        assert p.name == "test-gk"
        assert p.type == "ollama"
        assert p.ollama_host == "http://gpu-host:11434"
        assert p.model == "qwen2.5:3b"
        assert p.system_prompt == "You are a gatekeeper."
        assert p.temperature == 0.0
        assert p.num_predict == 24
        assert len(p.prompts) == 2
        assert p.classify is not None
        assert "approved" in p.classify
        assert "blocked" in p.classify

    def test_load_http_profile(self, tmp_path):
        path = self._write_yaml(tmp_path, """
name: test-http
type: http
model: qwen2.5:3b
http:
  url: "http://app-host:30500/classify"
  method: POST
  headers:
    Content-Type: application/json
    X-Custom: test
  body_template: '{"prompt": "{{ prompt }}", "model": "{{ model }}"}'
  timeout: 15
prompts:
  - "test prompt 1"
response:
  format: json
  text_field: decision
  latency_field: elapsed_ms
classify:
  approved: ["APPROVE"]
  blocked: ["DENY"]
""")
        p = load_profile(path)
        assert p.name == "test-http"
        assert p.type == "http"
        assert p.http_url == "http://app-host:30500/classify"
        assert p.http_method == "POST"
        assert p.http_headers["X-Custom"] == "test"
        assert p.http_timeout == 15
        assert p.response_text_field == "decision"
        assert p.response_latency_field == "elapsed_ms"

    def test_load_missing_file(self):
        with pytest.raises(ProfileError, match="not found"):
            load_profile("/nonexistent/profile.yaml")

    def test_load_invalid_yaml(self, tmp_path):
        path = self._write_yaml(tmp_path, "just a string")
        with pytest.raises(ProfileError, match="YAML mapping"):
            load_profile(path)

    def test_load_model_from_ollama_section(self, tmp_path):
        path = self._write_yaml(tmp_path, """
name: t
type: ollama
ollama:
  host: "http://h:11434"
  model: mistral:7b
""")
        p = load_profile(path)
        assert p.model == "mistral:7b"

    def test_top_level_model_overrides_ollama(self, tmp_path):
        path = self._write_yaml(tmp_path, """
name: t
type: ollama
model: qwen2.5:7b
ollama:
  host: "http://h:11434"
  model: qwen2.5:3b
""")
        p = load_profile(path)
        assert p.model == "qwen2.5:7b"

    def test_name_defaults_to_stem(self, tmp_path):
        f = tmp_path / "my-gate.yaml"
        f.write_text("""
type: ollama
ollama:
  host: "http://h:11434"
""")
        p = load_profile(str(f))
        assert p.name == "my-gate"

    def test_prompts_from_file(self, tmp_path):
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("# comment\nPrompt one\nPrompt two\n\n  \nPrompt three\n")
        path = self._write_yaml(tmp_path, f"""
name: t
type: ollama
ollama:
  host: "http://h:11434"
prompts_file: "{prompts_file}"
""")
        p = load_profile(path)
        assert p.prompts == ["Prompt one", "Prompt two", "Prompt three"]

    def test_prompts_file_relative(self, tmp_path):
        prompts_file = tmp_path / "my_prompts.txt"
        prompts_file.write_text("A\nB\n")
        path = self._write_yaml(tmp_path, """
name: t
type: ollama
ollama:
  host: "http://h:11434"
prompts_file: my_prompts.txt
""")
        p = load_profile(path)
        assert p.prompts == ["A", "B"]

    def test_inline_prompts_override_prompts_file(self, tmp_path):
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("from file")
        path = self._write_yaml(tmp_path, f"""
name: t
type: ollama
ollama:
  host: "http://h:11434"
prompts:
  - "inline prompt"
prompts_file: "{prompts_file}"
""")
        p = load_profile(path)
        assert p.prompts == ["inline prompt"]


# ── Template rendering ────────────────────────────────────────────────────────


class TestRenderBody:
    def test_basic_substitution(self):
        p = TargetProfile(
            name="t", type="http",
            http_url="http://h",
            http_body_template='{"prompt": "{{ prompt }}", "model": "{{ model }}"}',
            model="qwen2.5:3b",
        )
        result = render_body(p, "test prompt")
        parsed = json.loads(result)
        assert parsed["prompt"] == "test prompt"
        assert parsed["model"] == "qwen2.5:3b"

    def test_model_override(self):
        p = TargetProfile(
            name="t", type="http",
            http_url="http://h",
            http_body_template='{"model": "{{ model }}"}',
            model="default",
        )
        result = render_body(p, "x", model="override")
        assert json.loads(result)["model"] == "override"

    def test_num_predict(self):
        p = TargetProfile(
            name="t", type="http",
            http_url="http://h",
            http_body_template='{"n": {{ num_predict }}}',
        )
        result = render_body(p, "x", num_predict=64)
        assert "64" in result

    def test_no_template_returns_default(self):
        p = TargetProfile(
            name="t", type="http",
            http_url="http://h",
        )
        result = render_body(p, "hello")
        parsed = json.loads(result)
        assert parsed["prompt"] == "hello"

    def test_unknown_variable_preserved(self):
        p = TargetProfile(
            name="t", type="http",
            http_url="http://h",
            http_body_template='{{ unknown_var }}',
        )
        result = render_body(p, "x")
        assert "{{ unknown_var }}" in result

    def test_whitespace_in_braces(self):
        p = TargetProfile(
            name="t", type="http",
            http_url="http://h",
            http_body_template='{{  prompt  }}',
        )
        result = render_body(p, "spaced")
        assert result == "spaced"


# ── Classification ────────────────────────────────────────────────────────────


class TestClassifyResponse:
    def test_approved(self):
        p = TargetProfile(
            name="t", type="ollama", ollama_host="http://h:11434",
            classify={"approved": ["APPROVE", "ALLOW"], "blocked": ["DENY"]},
        )
        assert classify_response(p, "APPROVE: pod is safe") == "approved"

    def test_blocked(self):
        p = TargetProfile(
            name="t", type="ollama", ollama_host="http://h:11434",
            classify={"approved": ["APPROVE"], "blocked": ["DENY", "BLOCK"]},
        )
        assert classify_response(p, "DENY - dangerous image") == "blocked"

    def test_case_insensitive(self):
        p = TargetProfile(
            name="t", type="ollama", ollama_host="http://h:11434",
            classify={"approved": ["approve"]},
        )
        assert classify_response(p, "APPROVE") == "approved"

    def test_unknown_when_no_match(self):
        p = TargetProfile(
            name="t", type="ollama", ollama_host="http://h:11434",
            classify={"approved": ["YES"], "blocked": ["NO"]},
        )
        assert classify_response(p, "maybe later") == "unknown"

    def test_no_classify_rules(self):
        p = TargetProfile(
            name="t", type="ollama", ollama_host="http://h:11434",
        )
        assert classify_response(p, "anything") is None

    def test_empty_text(self):
        p = TargetProfile(
            name="t", type="ollama", ollama_host="http://h:11434",
            classify={"approved": ["YES"]},
        )
        assert classify_response(p, "") is None


# ── Text/latency extraction ──────────────────────────────────────────────────


class TestExtractText:
    def test_configured_field(self):
        p = TargetProfile(
            name="t", type="http", http_url="http://h",
            response_text_field="decision",
        )
        assert _extract_text(p, {"decision": "APPROVE"}) == "APPROVE"

    def test_auto_detect_response(self):
        p = TargetProfile(name="t", type="http", http_url="http://h")
        assert _extract_text(p, {"response": "hello"}) == "hello"

    def test_auto_detect_text(self):
        p = TargetProfile(name="t", type="http", http_url="http://h")
        assert _extract_text(p, {"text": "world"}) == "world"

    def test_fallback_to_json(self):
        p = TargetProfile(name="t", type="http", http_url="http://h")
        result = _extract_text(p, {"foo": "bar"})
        assert "foo" in result

    def test_non_dict(self):
        p = TargetProfile(name="t", type="http", http_url="http://h")
        assert _extract_text(p, "plain text") == "plain text"


class TestExtractLatency:
    def test_configured_field(self):
        p = TargetProfile(
            name="t", type="http", http_url="http://h",
            response_latency_field="elapsed_ms",
        )
        assert _extract_latency(p, {"elapsed_ms": 42.5}) == 42.5

    def test_no_field_configured(self):
        p = TargetProfile(name="t", type="http", http_url="http://h")
        assert _extract_latency(p, {"elapsed_ms": 42.5}) is None

    def test_missing_field(self):
        p = TargetProfile(
            name="t", type="http", http_url="http://h",
            response_latency_field="elapsed_ms",
        )
        assert _extract_latency(p, {"other": 10}) is None

    def test_non_numeric(self):
        p = TargetProfile(
            name="t", type="http", http_url="http://h",
            response_latency_field="elapsed_ms",
        )
        assert _extract_latency(p, {"elapsed_ms": "not a number"}) is None


# ── Async request execution (mocked) ─────────────────────────────────────────


class TestSingleRequestProfile:
    @pytest.mark.asyncio
    async def test_ollama_mode(self):
        from atomics.profiles import _single_request_profile

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "response": "APPROVE",
            "total_duration": 500_000_000,
            "eval_count": 10,
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        profile = TargetProfile(
            name="t", type="ollama",
            ollama_host="http://localhost:11434",
            model="qwen2.5:3b",
            system_prompt="You are a gate.",
            temperature=0.0,
            num_predict=24,
            classify={"approved": ["APPROVE"], "blocked": ["DENY"]},
        )

        text, latency, classification = await _single_request_profile(
            mock_client, profile, "Allow pod nginx"
        )

        assert text == "APPROVE"
        assert latency == pytest.approx(500.0)
        assert classification == "approved"

        call_args = mock_client.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert body["system"] == "You are a gate."
        assert body["options"]["temperature"] == 0.0
        assert body["options"]["num_predict"] == 24

    @pytest.mark.asyncio
    async def test_http_mode_json(self):
        from atomics.profiles import _single_request_profile

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "decision": "DENY",
            "elapsed_ms": 150.0,
        }

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)

        profile = TargetProfile(
            name="t", type="http",
            http_url="http://vm:5000/classify",
            http_body_template='{"prompt": "{{ prompt }}"}',
            http_timeout=10,
            response_format="json",
            response_text_field="decision",
            response_latency_field="elapsed_ms",
            classify={"approved": ["APPROVE"], "blocked": ["DENY"]},
        )

        text, latency, classification = await _single_request_profile(
            mock_client, profile, "test"
        )

        assert text == "DENY"
        assert latency == 150.0
        assert classification == "blocked"

    @pytest.mark.asyncio
    async def test_http_mode_text(self):
        from atomics.profiles import _single_request_profile

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = "OK: approved"

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_resp)

        profile = TargetProfile(
            name="t", type="http",
            http_url="http://vm:5000/gate",
            response_format="text",
            classify={"approved": ["approved"]},
        )

        text, latency, classification = await _single_request_profile(
            mock_client, profile, "test"
        )

        assert text == "OK: approved"
        assert classification == "approved"
        assert latency > 0

    @pytest.mark.asyncio
    async def test_http_get_method(self):
        from atomics.profiles import _single_request_profile

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = "pong"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        profile = TargetProfile(
            name="t", type="http",
            http_url="http://vm:5000/health",
            http_method="GET",
            response_format="text",
        )

        text, latency, _ = await _single_request_profile(
            mock_client, profile, "test"
        )

        assert text == "pong"
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_classification_returns_none(self):
        from atomics.profiles import _single_request_profile

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"response": "hello"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        profile = TargetProfile(
            name="t", type="ollama",
            ollama_host="http://localhost:11434",
            model="test",
        )

        _, _, classification = await _single_request_profile(
            mock_client, profile, "test"
        )

        assert classification is None
