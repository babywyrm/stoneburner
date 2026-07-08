"""Tests for atomics.validation — URL validation and error sanitization."""

from __future__ import annotations

import pytest

from atomics.validation import sanitize_error, validate_endpoint_url


class TestValidateEndpointUrl:
    def test_valid_http(self):
        assert validate_endpoint_url("http://192.168.1.239:11434") == "http://192.168.1.239:11434"

    def test_valid_https(self):
        assert validate_endpoint_url("https://api.example.com/v1/") == "https://api.example.com/v1"

    def test_strips_trailing_slash(self):
        assert validate_endpoint_url("http://host:8080/") == "http://host:8080"

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty URL"):
            validate_endpoint_url("")

    def test_rejects_file_scheme(self):
        with pytest.raises(ValueError, match="unsupported scheme"):
            validate_endpoint_url("file:///etc/passwd")

    def test_rejects_ftp_scheme(self):
        with pytest.raises(ValueError, match="unsupported scheme"):
            validate_endpoint_url("ftp://evil.com/data")

    def test_rejects_embedded_credentials(self):
        with pytest.raises(ValueError, match="embedded credentials"):
            validate_endpoint_url("http://user:pass@host:11434")

    def test_rejects_path_traversal(self):
        with pytest.raises(ValueError, match="path traversal"):
            validate_endpoint_url("http://host:11434/../../etc/passwd")

    def test_rejects_no_hostname(self):
        with pytest.raises(ValueError, match="missing hostname"):
            validate_endpoint_url("http://")

    def test_custom_label_in_error(self):
        with pytest.raises(ValueError, match="--ollama-host"):
            validate_endpoint_url("ftp://x", label="--ollama-host")


class TestSanitizeError:
    def test_strips_bearer_token(self):
        exc = Exception("Request failed: Bearer sk-ant-api03-abc123def456 in headers")
        result = sanitize_error(exc)
        assert "sk-ant-api03" not in result
        assert "[REDACTED]" in result

    def test_strips_sk_key(self):
        exc = Exception("Auth error with key sk-abcdef123456789012345")
        result = sanitize_error(exc)
        assert "sk-abcdef" not in result
        assert "[REDACTED]" in result

    def test_strips_github_token(self):
        exc = Exception("ghp_ABCDEFghijklmnopqrstuv12345 expired")
        result = sanitize_error(exc)
        assert "ghp_" not in result

    def test_strips_aws_key(self):
        exc = Exception("AKIAIOSFODNN7EXAMPLE was rejected")
        result = sanitize_error(exc)
        assert "AKIAIOSFODNN7" not in result

    def test_preserves_safe_content(self):
        exc = Exception("Connection refused to http://localhost:11434")
        result = sanitize_error(exc)
        assert "Connection refused" in result
        assert "localhost:11434" in result

    def test_truncates_long_messages(self):
        exc = Exception("x" * 1000)
        result = sanitize_error(exc)
        assert len(result) <= 510  # 500 + possible [REDACTED] expansion

    def test_handles_empty_str_exception(self):
        exc = ConnectionError()
        result = sanitize_error(exc)
        assert "ConnectionError" in result
