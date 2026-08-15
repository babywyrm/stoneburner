"""Normalized outcome of a single provider attempt.

This is the providers layer's own contract: classifying a finish reason, a
policy block, a rate limit, or a transport failure is the adapter's concern, not
the eval runner's. It lives here so `atomics.providers` no longer has to import
from `atomics.eval`. `atomics.eval.outcomes` re-exports every public name below,
so the historical import path keeps working.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx

from atomics.validation import sanitize_error


class ProviderOutcomeKind(StrEnum):
    """Normalized outcome of one provider attempt."""

    COMPLETED = "completed"
    REFUSED = "refused"
    SAFETY_BLOCKED = "safety_blocked"
    TRUNCATED = "truncated"
    EMPTY = "empty"
    THINKING_BUDGET = "thinking_budget"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"
    TRANSPORT_ERROR = "transport_error"


_SCORABLE_PROVIDER_OUTCOMES = frozenset(
    {
        ProviderOutcomeKind.COMPLETED,
        ProviderOutcomeKind.REFUSED,
        ProviderOutcomeKind.SAFETY_BLOCKED,
        ProviderOutcomeKind.TRUNCATED,
    }
)
_INFRASTRUCTURE_INVALID_PROVIDER_OUTCOMES = frozenset(
    {
        ProviderOutcomeKind.RATE_LIMITED,
        ProviderOutcomeKind.TIMEOUT,
        ProviderOutcomeKind.PROVIDER_ERROR,
        ProviderOutcomeKind.TRANSPORT_ERROR,
    }
)


@dataclass(frozen=True)
class ProviderOutcome:
    """Normalized provider result and optional diagnostic details."""

    kind: ProviderOutcomeKind
    finish_reason: str | None = None
    safety_reason: str | None = None
    error_class: str | None = None
    error_message: str | None = None

    @property
    def is_scorable(self) -> bool:
        return self.kind in _SCORABLE_PROVIDER_OUTCOMES

    @property
    def is_infrastructure_invalid(self) -> bool:
        return self.kind in _INFRASTRUCTURE_INVALID_PROVIDER_OUTCOMES


def provider_outcome_from_exception(exc: BaseException) -> ProviderOutcome:
    """Normalize a provider exception without persisting secrets."""

    response = getattr(exc, "response", None)
    if isinstance(exc, httpx.TimeoutException):
        kind = ProviderOutcomeKind.TIMEOUT
    elif _is_rate_limit_exception(exc):
        kind = ProviderOutcomeKind.RATE_LIMITED
    elif isinstance(exc, httpx.TransportError):
        kind = ProviderOutcomeKind.TRANSPORT_ERROR
    elif safety_reason := policy_block_reason(
        code=getattr(exc, "code", None),
        message=(
            getattr(exc, "message", None) or str(exc)
            if _has_structured_provider_context(exc)
            else None
        ),
        body=(
            getattr(exc, "body", None),
            {
                "code": getattr(response, "code", None),
                "message": getattr(response, "message", None),
            },
            getattr(response, "body", None),
            _structured_response_json(response),
        ),
    ):
        kind = ProviderOutcomeKind.SAFETY_BLOCKED
    else:
        kind = ProviderOutcomeKind.PROVIDER_ERROR

    return ProviderOutcome(
        kind=kind,
        safety_reason=safety_reason if kind is ProviderOutcomeKind.SAFETY_BLOCKED else None,
        error_class=type(exc).__name__,
        error_message=sanitize_error(exc),
    )


def _is_rate_limit_exception(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return status_code == 429 or response_status == 429 or "ratelimit" in type(exc).__name__.lower()


def _has_structured_provider_context(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    return (
        isinstance(exc, httpx.HTTPStatusError)
        or isinstance(getattr(exc, "status_code", None), int)
        or isinstance(getattr(response, "status_code", None), int)
        or getattr(exc, "body", None) is not None
        or getattr(exc, "code", None) is not None
    )


def _structured_response_json(response: object) -> object:
    if response is None:
        return None
    json_method = getattr(response, "json", None)
    if callable(json_method):
        try:
            parsed = json_method()
        except (TypeError, ValueError):
            pass
        else:
            return parsed if isinstance(parsed, (dict, list)) else None

    text = getattr(response, "text", None)
    if not isinstance(text, str):
        return None
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, (dict, list)) else None


_POLICY_CODE_ALIASES = {
    "content_filter": "content_filter",
    "content_filtered": "content_filter",
    "content_policy_violation": "content_policy_violation",
    "content_policy_violated": "content_policy_violation",
    "content_policy_blocked": "content_policy_violation",
    "image_content_policy_violation": "image_content_policy_violation",
    "image_content_policy_blocked": "image_content_policy_violation",
    "image_policy_violation": "image_content_policy_violation",
    "safety_policy_violation": "policy_block",
    "responsible_ai_policy_violation": "policy_block",
    "responsibleaipolicyviolation": "policy_block",
    "policy_block": "policy_block",
    "policy_blocked": "policy_block",
    "blocked_by_policy": "policy_block",
    "cybersecurity_risk": "cybersecurity_risk",
    "cyber_security_risk": "cybersecurity_risk",
}
_POLICY_MESSAGE_PATTERNS = (
    (
        re.compile(r"\bblocked by (?:the )?(?:safety|content) policy\b", re.IGNORECASE),
        "policy_block",
    ),
    (
        re.compile(r"\bviolat(?:e|es|ed|ing) (?:the )?content policy\b", re.IGNORECASE),
        "content_policy_violation",
    ),
    (
        re.compile(r"\bcybersecurity risk\b", re.IGNORECASE),
        "cybersecurity_risk",
    ),
)
_NEGATED_POLICY_MESSAGE = re.compile(
    r"\b(?:not|never)\s+blocked by (?:the )?(?:safety|content) policy\b"
    r"|\b(?:did|does|do)\s+not\s+violate (?:the )?content policy\b",
    re.IGNORECASE,
)


def policy_block_reason(
    *,
    code: object = None,
    message: object = None,
    body: object = None,
) -> str | None:
    """Return a canonical policy reason from structured codes or explicit prose."""

    codes: list[str] = []
    messages: list[str] = []
    if code is not None:
        codes.append(str(code))
    if message is not None:
        messages.append(str(message))
    _collect_policy_fields(body, codes=codes, messages=messages)

    for candidate in codes:
        normalized = re.sub(r"[^a-z0-9]+", "_", candidate.lower()).strip("_")
        if canonical := _POLICY_CODE_ALIASES.get(normalized):
            return canonical

    for candidate in messages:
        if _NEGATED_POLICY_MESSAGE.search(candidate):
            continue
        for pattern, canonical in _POLICY_MESSAGE_PATTERNS:
            if pattern.search(candidate):
                return canonical
    return None


def _collect_policy_fields(
    value: Any,
    *,
    codes: list[str],
    messages: list[str],
) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = str(key).lower()
            if normalized_key in {"code", "reason", "type"} and isinstance(
                nested, (str, int)
            ):
                codes.append(str(nested))
            elif normalized_key in {"message", "detail", "error_description"} and isinstance(
                nested, str
            ):
                messages.append(nested)
            if isinstance(nested, (dict, list, tuple)):
                _collect_policy_fields(nested, codes=codes, messages=messages)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _collect_policy_fields(nested, codes=codes, messages=messages)
