"""Input validation helpers for CLI and config boundaries."""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Collection
from dataclasses import dataclass
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
        raise ValueError(f"{label}: unsupported scheme {parsed.scheme!r} — only http/https allowed")

    if parsed.username or parsed.password:
        raise ValueError(f"{label}: embedded credentials not allowed in endpoint URLs")

    if not parsed.hostname:
        raise ValueError(f"{label}: missing hostname")

    if ".." in parsed.path:
        raise ValueError(f"{label}: path traversal ('..') not allowed")

    return url


@dataclass(frozen=True)
class AllowedEndpoint:
    """Normalized outbound HTTP(S) origin plus optional path boundary."""

    scheme: str
    host: str
    port: int
    path_prefix: str = ""


def parse_allowed_endpoint(url: str, *, label: str = "endpoint") -> AllowedEndpoint:
    """Validate and canonicalize an HTTP(S) endpoint URL for origin/path matching."""
    raw = url.strip()
    if "?" in raw or "#" in raw:
        raise ValueError(f"{label}: query strings and fragments are not allowed in endpoint URLs")
    cleaned = validate_endpoint_url(raw, label=label)
    parsed = urlparse(cleaned)
    try:
        explicit_port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label}: invalid port in endpoint URL") from exc
    port = explicit_port if explicit_port is not None else (443 if parsed.scheme == "https" else 80)
    if port < 1 or port > 65535:
        raise ValueError(f"{label}: port out of range 1..65535")

    raw_host = (parsed.hostname or "").lower().rstrip(".")
    if not raw_host:
        raise ValueError(f"{label}: missing hostname")
    try:
        ip = ipaddress.ip_address(raw_host)
    except ValueError:
        canonical_host = raw_host
    else:
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        canonical_host = str(ip)

    path_prefix = parsed.path.rstrip("/")
    return AllowedEndpoint(
        scheme=parsed.scheme,
        host=canonical_host,
        port=port,
        path_prefix=path_prefix,
    )


def is_endpoint_allowed(
    url: str,
    allowed: Collection[AllowedEndpoint],
    *,
    label: str = "host",
) -> bool:
    """Return True if `url` matches one of the normalized `allowed` origins/prefixes."""
    candidate = parse_allowed_endpoint(url, label=label)
    for rule in allowed:
        if (
            candidate.scheme != rule.scheme
            or candidate.host != rule.host
            or candidate.port != rule.port
        ):
            continue
        if not rule.path_prefix:
            return True
        if candidate.path_prefix == rule.path_prefix or candidate.path_prefix.startswith(
            rule.path_prefix + "/"
        ):
            return True
    return False


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
    return f"{match.group('prefix')}{match.group('quote')}[REDACTED]{match.group('quote')}"


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
