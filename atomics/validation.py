"""Input validation helpers for CLI and config boundaries."""

from __future__ import annotations

import re
from urllib.parse import urlparse


def validate_endpoint_url(url: str, *, label: str = "URL") -> str:
    """Validate and normalize an HTTP(S) endpoint URL.

    Rejects file://, embedded credentials, path traversal, and non-HTTP schemes.
    Returns the cleaned URL (trailing slash stripped).

    Raises ValueError with a user-friendly message on invalid input.
    """
    url = url.strip().rstrip("/")
    if not url:
        raise ValueError(f"{label}: empty URL")

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"{label}: unsupported scheme {parsed.scheme!r} — only http/https allowed"
        )

    if parsed.username or parsed.password:
        raise ValueError(
            f"{label}: embedded credentials not allowed in endpoint URLs"
        )

    if not parsed.hostname:
        raise ValueError(f"{label}: missing hostname")

    if ".." in parsed.path:
        raise ValueError(f"{label}: path traversal ('..') not allowed")

    return url


_TOKEN_PATTERNS = re.compile(
    r"(Bearer\s+\S+|sk-[a-zA-Z0-9_-]{10,}|ghp_[a-zA-Z0-9]{20,}|AKIA[A-Z0-9]{16})",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT_PREFIX = r"""
    (?P<prefix>
        (?<![a-z0-9_-])
        ["']?
        (?:
            password
            |passwd
            |api_key
            |access_token
            |refresh_token
            |client_secret
            |aws_secret_access_key
            |aws_session_token
            |x-api-key
            |x_api_key
            |x-amz-signature
            |x-amz-credential
            |x-amz-security-token
        )
        ["']?
        (?![a-z0-9_-])
        \s*(?:=|:)\s*
    )
"""
_QUOTED_SECRET_PATTERN = re.compile(
    _SECRET_ASSIGNMENT_PREFIX
    + r"""
    (?P<quote>["'])
    (?P<value>(?:\\.|(?!(?P=quote))[^\\])*)
    (?P=quote)
    """,
    re.IGNORECASE | re.VERBOSE,
)
_UNQUOTED_SECRET_PATTERN = re.compile(
    _SECRET_ASSIGNMENT_PREFIX
    + r"""
    (?P<value>[^"';&,\n}\]]*?\S)
    (?P<trailing>\s*)
    (?=[;&,\n}\]]|$)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _redact_quoted_secret(match: re.Match[str]) -> str:
    return (
        f"{match.group('prefix')}{match.group('quote')}"
        f"[REDACTED]{match.group('quote')}"
    )


def _redact_unquoted_secret(match: re.Match[str]) -> str:
    return f"{match.group('prefix')}[REDACTED]{match.group('trailing')}"


def sanitize_error(exc: BaseException) -> str:
    """Return a sanitized error string safe for DB persistence and log output.

    Strips common secret patterns (Bearer tokens, API keys) from the exception
    text to prevent accidental credential leakage into stored results or exports.
    """
    raw = str(exc) or repr(exc)
    sanitized = _QUOTED_SECRET_PATTERN.sub(_redact_quoted_secret, raw)
    sanitized = _UNQUOTED_SECRET_PATTERN.sub(_redact_unquoted_secret, sanitized)
    sanitized = _TOKEN_PATTERNS.sub("[REDACTED]", sanitized)
    return sanitized[:500]
