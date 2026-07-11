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

    def test_strips_password_assignment(self):
        result = sanitize_error(Exception("Login failed: password=hunter2"))
        assert result == "Login failed: password=[REDACTED]"

    def test_strips_json_api_key_assignment(self):
        result = sanitize_error(
            Exception('Request body: {"api_key": "ordinary-looking-secret"}')
        )
        assert result == 'Request body: {"api_key": "[REDACTED]"}'

    def test_strips_full_quoted_multiword_secret(self):
        result = sanitize_error(
            Exception('client_secret: "two words", status: "denied"')
        )
        assert result == 'client_secret: "[REDACTED]", status: "denied"'

    def test_strips_quoted_secret_with_escaped_quote(self):
        result = sanitize_error(
            Exception('{"api_key":"abc\\\"def-secret","status":"denied"}')
        )
        assert result == '{"api_key":"[REDACTED]","status":"denied"}'

    def test_strips_delimiter_bounded_unquoted_multiword_secret(self):
        result = sanitize_error(
            Exception("password=correct horse battery staple; status=denied")
        )
        assert result == "password=[REDACTED]; status=denied"

    def test_strips_cloud_credentials_from_signed_s3_url(self):
        message = (
            "GET https://bucket.s3.amazonaws.com/object?"
            "X-Amz-Algorithm=AWS4-HMAC-SHA256"
            "&X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F20260711%2Fus-east-1%2Fs3"
            "&X-Amz-Signature=deadbeefcafebabe"
            "&X-Amz-Security-Token=session-token-secret"
            "&response-content-type=text%2Fplain"
        )

        result = sanitize_error(Exception(message))

        assert "AKIAIOSFODNN7EXAMPLE%2F20260711" not in result
        assert "%2F20260711%2Fus-east-1%2Fs3" not in result
        assert "deadbeefcafebabe" not in result
        assert "session-token-secret" not in result
        assert "X-Amz-Algorithm=AWS4-HMAC-SHA256" in result
        assert "response-content-type=text%2Fplain" in result
        assert result.count("[REDACTED]") == 3

    def test_strips_cloud_credentials_from_headers_and_environment(self):
        message = (
            'X-API-KEY: "abc\\\"def-secret"\n'
            "AWS_SECRET_ACCESS_KEY=aws-secret-value\n"
            "aws_session_token: session-token-value\n"
            "status: denied"
        )

        result = sanitize_error(Exception(message))

        assert "abc" not in result
        assert "def-secret" not in result
        assert "aws-secret-value" not in result
        assert "session-token-value" not in result
        assert "status: denied" in result
        assert result.count("[REDACTED]") == 3

    def test_strips_underscore_api_header_key(self):
        result = sanitize_error(Exception("x_api_key=header-secret; status=denied"))

        assert result == "x_api_key=[REDACTED]; status=denied"

    def test_redacts_secret_before_truncating_message(self):
        secret = "start-secret-" + ("z" * 80) + "-end-secret"
        message = ("x" * 480) + f' api_key="{secret}"'

        result = sanitize_error(Exception(message))

        assert "start-secret" not in result
        assert "[REDACTED]" in result
        assert len(result) <= 500

    def test_preserves_ordinary_secret_related_prose(self):
        message = "The password policy and API key documentation are unavailable"
        assert sanitize_error(Exception(message)) == message

    def test_preserves_cloud_credential_key_names_in_prose(self):
        message = (
            "Rotate AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, and x-api-key regularly"
        )
        assert sanitize_error(Exception(message)) == message

    def test_does_not_redact_larger_identifier_suffix(self):
        message = "notpassword=hunter2 is a harmless diagnostic field"
        assert sanitize_error(Exception(message)) == message

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
