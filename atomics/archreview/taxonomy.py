"""Fixed OWASP-style security category taxonomy for archreview findings.

One shared enum so findings are comparable across repos and models. A repo
spec may later select a different taxonomy; v1 ships this web-app-oriented set.
"""

from __future__ import annotations

import re
from enum import StrEnum


class Category(StrEnum):
    BROKEN_ACCESS_CONTROL = "broken_access_control"
    BROKEN_AUTHENTICATION = "broken_authentication"
    INJECTION = "injection"
    XSS = "xss"
    XXE = "xxe"
    CRYPTOGRAPHIC_ISSUES = "cryptographic_issues"
    SENSITIVE_DATA_EXPOSURE = "sensitive_data_exposure"
    IMPROPER_INPUT_VALIDATION = "improper_input_validation"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    VULNERABLE_COMPONENTS = "vulnerable_components"
    SECURITY_MISCONFIGURATION = "security_misconfiguration"
    SSRF = "ssrf"
    BROKEN_ANTI_AUTOMATION = "broken_anti_automation"
    MISCELLANEOUS = "miscellaneous"


ALL_CATEGORIES: tuple[Category, ...] = tuple(Category)

# Synonym phrases → canonical category. Keys are lowercased; matching is done
# against a lowercased, non-alphanumeric-collapsed form of the input.
_SYNONYMS: dict[str, Category] = {
    "broken access control": Category.BROKEN_ACCESS_CONTROL,
    "access control": Category.BROKEN_ACCESS_CONTROL,
    "idor": Category.BROKEN_ACCESS_CONTROL,
    "authorization": Category.BROKEN_ACCESS_CONTROL,
    "broken authentication": Category.BROKEN_AUTHENTICATION,
    "authentication": Category.BROKEN_AUTHENTICATION,
    "auth bypass": Category.BROKEN_AUTHENTICATION,
    "injection": Category.INJECTION,
    "sql injection": Category.INJECTION,
    "sqli": Category.INJECTION,
    "command injection": Category.INJECTION,
    "nosql injection": Category.INJECTION,
    "xss": Category.XSS,
    "cross site scripting": Category.XSS,
    "xxe": Category.XXE,
    "xml external entity": Category.XXE,
    "cryptographic issues": Category.CRYPTOGRAPHIC_ISSUES,
    "crypto": Category.CRYPTOGRAPHIC_ISSUES,
    "weak crypto": Category.CRYPTOGRAPHIC_ISSUES,
    "sensitive data exposure": Category.SENSITIVE_DATA_EXPOSURE,
    "information disclosure": Category.SENSITIVE_DATA_EXPOSURE,
    "improper input validation": Category.IMPROPER_INPUT_VALIDATION,
    "input validation": Category.IMPROPER_INPUT_VALIDATION,
    "insecure deserialization": Category.INSECURE_DESERIALIZATION,
    "deserialization": Category.INSECURE_DESERIALIZATION,
    "vulnerable components": Category.VULNERABLE_COMPONENTS,
    "vulnerable dependencies": Category.VULNERABLE_COMPONENTS,
    "outdated components": Category.VULNERABLE_COMPONENTS,
    "security misconfiguration": Category.SECURITY_MISCONFIGURATION,
    "misconfiguration": Category.SECURITY_MISCONFIGURATION,
    "ssrf": Category.SSRF,
    "server side request forgery": Category.SSRF,
    "broken anti automation": Category.BROKEN_ANTI_AUTOMATION,
    "anti automation": Category.BROKEN_ANTI_AUTOMATION,
    "rate limiting": Category.BROKEN_ANTI_AUTOMATION,
    "miscellaneous": Category.MISCELLANEOUS,
    "misc": Category.MISCELLANEOUS,
    # ── Non-web / cross-domain synonyms ─────────────────────────────────────
    # Access control / traversal
    "path traversal": Category.BROKEN_ACCESS_CONTROL,
    "directory traversal": Category.BROKEN_ACCESS_CONTROL,
    "lfi": Category.BROKEN_ACCESS_CONTROL,
    "rfi": Category.BROKEN_ACCESS_CONTROL,
    "local file inclusion": Category.BROKEN_ACCESS_CONTROL,
    "remote file inclusion": Category.BROKEN_ACCESS_CONTROL,
    "privilege escalation": Category.BROKEN_ACCESS_CONTROL,
    "open redirect": Category.BROKEN_ACCESS_CONTROL,
    "unvalidated redirect": Category.BROKEN_ACCESS_CONTROL,
    # Sensitive data / credentials
    "hardcoded credentials": Category.SENSITIVE_DATA_EXPOSURE,
    "hardcoded secret": Category.SENSITIVE_DATA_EXPOSURE,
    "hardcoded password": Category.SENSITIVE_DATA_EXPOSURE,
    "credential exposure": Category.SENSITIVE_DATA_EXPOSURE,
    "secret exposure": Category.SENSITIVE_DATA_EXPOSURE,
    "api key exposure": Category.SENSITIVE_DATA_EXPOSURE,
    # Injection variants
    "log injection": Category.INJECTION,
    "log forging": Category.INJECTION,
    "header injection": Category.INJECTION,
    "crlf injection": Category.INJECTION,
    "template injection": Category.INJECTION,
    "ssti": Category.INJECTION,
    "xpath injection": Category.INJECTION,
    "ldap injection": Category.INJECTION,
    "expression language injection": Category.INJECTION,
    # Crypto
    "weak cipher": Category.CRYPTOGRAPHIC_ISSUES,
    "weak hash": Category.CRYPTOGRAPHIC_ISSUES,
    "insecure random": Category.CRYPTOGRAPHIC_ISSUES,
    "weak random": Category.CRYPTOGRAPHIC_ISSUES,
    "insufficient entropy": Category.CRYPTOGRAPHIC_ISSUES,
    "tls misconfiguration": Category.SECURITY_MISCONFIGURATION,
    "ssl misconfiguration": Category.SECURITY_MISCONFIGURATION,
    "certificate validation": Category.CRYPTOGRAPHIC_ISSUES,
    # Components / supply chain
    "dependency confusion": Category.VULNERABLE_COMPONENTS,
    "supply chain": Category.VULNERABLE_COMPONENTS,
    "typosquatting": Category.VULNERABLE_COMPONENTS,
    "outdated library": Category.VULNERABLE_COMPONENTS,
    "known vulnerability": Category.VULNERABLE_COMPONENTS,
    # Input validation / logic
    "race condition": Category.IMPROPER_INPUT_VALIDATION,
    "toctou": Category.IMPROPER_INPUT_VALIDATION,
    "integer overflow": Category.IMPROPER_INPUT_VALIDATION,
    "integer underflow": Category.IMPROPER_INPUT_VALIDATION,
    "buffer overflow": Category.IMPROPER_INPUT_VALIDATION,
    "type confusion": Category.IMPROPER_INPUT_VALIDATION,
    "null dereference": Category.IMPROPER_INPUT_VALIDATION,
    "unhandled exception": Category.IMPROPER_INPUT_VALIDATION,
    # Automation / DoS
    "denial of service": Category.BROKEN_ANTI_AUTOMATION,
    "dos": Category.BROKEN_ANTI_AUTOMATION,
    "resource exhaustion": Category.BROKEN_ANTI_AUTOMATION,
    "regex dos": Category.BROKEN_ANTI_AUTOMATION,
    "redos": Category.BROKEN_ANTI_AUTOMATION,
}


def _canonical(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def normalize_category(raw: str) -> Category | None:
    """Map a free-text category label to a Category, or None if unmappable."""
    if not raw or not raw.strip():
        return None
    canon = _canonical(raw)
    if not canon:
        return None
    underscored = canon.replace(" ", "_")
    for cat in Category:
        if cat.value == underscored:
            return cat
    if canon in _SYNONYMS:
        return _SYNONYMS[canon]
    # Substring fallback: longest synonym phrase contained in the input wins.
    best: tuple[int, Category] | None = None
    for phrase, cat in _SYNONYMS.items():
        if phrase in canon and (best is None or len(phrase) > best[0]):
            best = (len(phrase), cat)
    return best[1] if best else None
