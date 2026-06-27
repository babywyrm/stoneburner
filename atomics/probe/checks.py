"""Probe checks — build LLM analysis tasks from artifact content.

Each handler receives the raw artifact content and returns a dict with:
    check_id:     str    — unique ID for this check type
    prompt:       str    — the full prompt to send to the analysis model
    gold_criteria: list  — what a good analysis should cover (for the judge)
"""

from __future__ import annotations

from collections.abc import Callable

_HANDLERS: dict[str, Callable[[str], dict]] = {}


def _register(artifact_type: str):
    def decorator(fn: Callable[[str], dict]):
        _HANDLERS[artifact_type] = fn
        return fn
    return decorator


def build_check(artifact_type: str, content: str) -> dict:
    """Build a check task dict for the given artifact type and content."""
    handler = _HANDLERS.get(artifact_type)
    if handler is None:
        return {
            "check_id": "generic_analysis",
            "prompt": (
                f"Analyse the following {artifact_type} artifact for security issues, "
                "anomalies, or misconfigurations. Summarise your findings clearly.\n\n"
                f"CONTENT:\n{content[:4000]}"
            ),
            "gold_criteria": [
                "identifies any security issues or anomalies present",
                "provides actionable recommendations",
            ],
        }
    return handler(content)


@_register("access-log")
def _access_log(content: str) -> dict:
    return {
        "check_id": "ioc_analysis",
        "prompt": (
            "You are a SOC analyst reviewing an nginx/Apache access log.\n\n"
            "Analyse the following log lines for Indicators of Compromise (IoCs), "
            "suspicious patterns, scanning activity, or attack attempts. "
            "For each suspicious finding:\n"
            "1. Identify the source IP(s) involved.\n"
            "2. Classify the attack type (e.g., SQL injection, path traversal, recon).\n"
            "3. Assess the severity (LOW/MEDIUM/HIGH/CRITICAL).\n"
            "4. Recommend an immediate defensive action.\n\n"
            f"LOG EXCERPT:\n{content[:4000]}"
        ),
        "gold_criteria": [
            "identifies suspicious IP addresses or patterns",
            "classifies at least one attack type correctly",
            "provides severity assessment for each finding",
            "recommends concrete defensive actions (firewall rule, WAF, block list)",
        ],
    }


@_register("json-security-report")
def _json_security_report(content: str) -> dict:
    return {
        "check_id": "finding_triage",
        "prompt": (
            "You are a security engineer triaging a JSON-format security scan report.\n\n"
            "Analyse the following findings. For each CRITICAL or HIGH severity finding:\n"
            "1. Summarise the vulnerability and its impact.\n"
            "2. Provide a remediation priority (immediate/within-sprint/backlog).\n"
            "3. Suggest a concrete remediation step.\n\n"
            "Also identify any false-positive candidates and explain your reasoning.\n\n"
            f"REPORT:\n{content[:4000]}"
        ),
        "gold_criteria": [
            "summarises all CRITICAL and HIGH findings",
            "assigns remediation priority to each high-severity finding",
            "provides at least one concrete remediation step",
            "notes any likely false positives",
        ],
    }


@_register("inference-api")
def _inference_api(content: str) -> dict:
    return {
        "check_id": "inference_api_health",
        "prompt": (
            "You are reviewing the response from an inference API endpoint (e.g., Ollama, vLLM).\n\n"
            "Assess the following API response for:\n"
            "1. Service health — is it responding correctly?\n"
            "2. Exposed models — are any unexpectedly exposed or outdated?\n"
            "3. Security posture — any unauthenticated access indicators, unusual model names, "
            "or configuration issues?\n"
            "4. Recommendations for hardening the inference endpoint.\n\n"
            f"API RESPONSE:\n{content[:4000]}"
        ),
        "gold_criteria": [
            "assesses whether the API is healthy",
            "lists exposed models and flags any unexpected ones",
            "identifies unauthenticated access if present",
            "provides at least one hardening recommendation",
        ],
    }


@_register("k8s-audit-log")
def _k8s_audit_log(content: str) -> dict:
    return {
        "check_id": "k8s_audit_anomaly",
        "prompt": (
            "You are a Kubernetes security engineer reviewing audit log events.\n\n"
            "Analyse the following Kubernetes audit log entries for:\n"
            "1. Privilege escalation attempts (e.g., ClusterRoleBinding creation, exec into pods).\n"
            "2. Unusual API calls (unexpected verbs, resources, or namespaces).\n"
            "3. Service account abuse or token requests.\n"
            "4. Any events that should trigger a security alert.\n\n"
            "For each finding, cite the relevant log entry and explain the risk.\n\n"
            f"AUDIT LOG:\n{content[:4000]}"
        ),
        "gold_criteria": [
            "identifies privilege escalation attempts or unusual API calls",
            "cites specific log entries for each finding",
            "explains the risk of each finding",
            "recommends alerting rules or admission policies",
        ],
    }


@_register("config-file")
def _config_file(content: str) -> dict:
    return {
        "check_id": "config_security_review",
        "prompt": (
            "You are a security engineer reviewing a configuration file.\n\n"
            "Analyse the following config for security misconfigurations, including:\n"
            "1. Hardcoded secrets, passwords, or tokens.\n"
            "2. Debug or verbose modes enabled in production.\n"
            "3. Overly permissive settings (e.g., CORS *, open ports, disabled auth).\n"
            "4. Missing security controls (encryption, TLS, auth).\n\n"
            "For each issue, indicate severity and a recommended fix.\n\n"
            f"CONFIG:\n{content[:4000]}"
        ),
        "gold_criteria": [
            "identifies hardcoded secrets or credentials",
            "flags debug/development settings",
            "identifies overly permissive settings",
            "provides severity and remediation for each issue",
        ],
    }


@_register("api-response")
def _api_response(content: str) -> dict:
    return {
        "check_id": "api_response_security",
        "prompt": (
            "You are a security engineer reviewing an API response for security implications.\n\n"
            "Analyse the following API response for:\n"
            "1. Sensitive data exposure (PII, internal paths, stack traces, credentials).\n"
            "2. Information disclosure (software versions, server details, internal IPs).\n"
            "3. Error handling quality — are errors exposing too much detail?\n"
            "4. Unexpected fields or data that should not be in the response.\n\n"
            f"API RESPONSE:\n{content[:4000]}"
        ),
        "gold_criteria": [
            "identifies any sensitive data in the response",
            "flags information disclosure issues",
            "assesses error handling quality",
            "recommends response filtering or data masking if needed",
        ],
    }
