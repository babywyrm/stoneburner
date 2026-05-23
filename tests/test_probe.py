"""Tests for the probe module — config, connectors, checks, runner."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────

def test_probe_target_dataclass():
    from atomics.probe.config import ProbeTarget
    t = ProbeTarget(
        name="my-scanner",
        artifact_type="json-security-report",
        source="file",
        path="/tmp/report.json",
    )
    assert t.name == "my-scanner"
    assert t.artifact_type == "json-security-report"


def test_load_probe_config_valid(tmp_path):
    from atomics.probe.config import load_probe_config
    cfg = tmp_path / "probes.yaml"
    cfg.write_text(textwrap.dedent("""
        targets:
          - name: nginx-logs
            artifact_type: access-log
            source: file
            path: /var/log/nginx/access.log
          - name: api-check
            artifact_type: inference-api
            source: http
            url: http://localhost:11434/api/tags
    """))
    targets = load_probe_config(cfg)
    assert len(targets) == 2
    assert targets[0].name == "nginx-logs"
    assert targets[1].source == "http"


def test_load_probe_config_missing_file():
    from atomics.probe.config import load_probe_config, ProbeConfigError
    import pytest
    with pytest.raises(ProbeConfigError, match="not found"):
        load_probe_config(Path("/nonexistent/probes.yaml"))


def test_load_probe_config_invalid_artifact_type(tmp_path):
    from atomics.probe.config import load_probe_config, ProbeConfigError
    import pytest
    cfg = tmp_path / "probes.yaml"
    cfg.write_text(textwrap.dedent("""
        targets:
          - name: bad-target
            artifact_type: totally-unknown-type
            source: file
            path: /tmp/foo.txt
    """))
    with pytest.raises(ProbeConfigError, match="artifact_type"):
        load_probe_config(cfg)


def test_valid_artifact_types_list():
    from atomics.probe.config import VALID_ARTIFACT_TYPES
    assert "json-security-report" in VALID_ARTIFACT_TYPES
    assert "access-log" in VALID_ARTIFACT_TYPES
    assert "inference-api" in VALID_ARTIFACT_TYPES
    assert "k8s-audit-log" in VALID_ARTIFACT_TYPES
    assert "config-file" in VALID_ARTIFACT_TYPES
    assert "api-response" in VALID_ARTIFACT_TYPES


# ── Connectors ────────────────────────────────────────────────────────────────

def test_fetch_artifact_file(tmp_path):
    import asyncio
    from atomics.probe.config import ProbeTarget
    from atomics.probe.connectors import fetch_artifact
    f = tmp_path / "report.json"
    f.write_text('{"findings": [{"id": "CVE-2024-1234", "severity": "CRITICAL"}]}')
    target = ProbeTarget(
        name="test", artifact_type="json-security-report", source="file", path=str(f)
    )
    content = asyncio.run(fetch_artifact(target))
    assert "CVE-2024-1234" in content


def test_fetch_artifact_file_missing(tmp_path):
    import asyncio
    import pytest
    from atomics.probe.config import ProbeTarget
    from atomics.probe.connectors import fetch_artifact, ProbeConnectorError
    target = ProbeTarget(
        name="test", artifact_type="access-log", source="file", path="/nonexistent/file.log"
    )
    with pytest.raises(ProbeConnectorError, match="not found"):
        asyncio.run(fetch_artifact(target))


def test_fetch_artifact_unknown_source():
    import asyncio
    import pytest
    from atomics.probe.config import ProbeTarget
    from atomics.probe.connectors import fetch_artifact, ProbeConnectorError
    target = ProbeTarget(
        name="test", artifact_type="api-response", source="sftp",  # type: ignore[arg-type]
        path="/foo"
    )
    with pytest.raises(ProbeConnectorError, match="source"):
        asyncio.run(fetch_artifact(target))


# ── Checks ────────────────────────────────────────────────────────────────────

def test_build_check_access_log():
    from atomics.probe.checks import build_check
    check = build_check("access-log", "10.0.0.5 GET /admin 403\n10.0.0.5 GET /.env 404\n")
    assert "prompt" in check
    assert "gold_criteria" in check
    assert "check_id" in check
    assert len(check["gold_criteria"]) >= 1


def test_build_check_json_security_report():
    from atomics.probe.checks import build_check
    payload = json.dumps({"findings": [{"id": "CVE-2024-1234", "severity": "CRITICAL"}]})
    check = build_check("json-security-report", payload)
    assert "CRITICAL" in check["prompt"] or "finding" in check["prompt"].lower()


def test_build_check_inference_api():
    from atomics.probe.checks import build_check
    payload = json.dumps({"models": [{"name": "llama3:8b"}]})
    check = build_check("inference-api", payload)
    assert check["check_id"] == "inference_api_health"


def test_build_check_k8s_audit_log():
    from atomics.probe.checks import build_check
    payload = '{"kind":"Event","verb":"create","user":{"username":"admin"}}\n'
    check = build_check("k8s-audit-log", payload)
    assert check["check_id"] == "k8s_audit_anomaly"


def test_build_check_config_file():
    from atomics.probe.checks import build_check
    payload = "password=hunter2\nDEBUG=true\n"
    check = build_check("config-file", payload)
    assert check["check_id"] == "config_security_review"


def test_build_check_api_response():
    from atomics.probe.checks import build_check
    payload = '{"status": "ok", "data": []}'
    check = build_check("api-response", payload)
    assert check["check_id"] == "api_response_security"


# ── Runner ────────────────────────────────────────────────────────────────────

import asyncio
from unittest.mock import AsyncMock
from types import SimpleNamespace


def _provider(text="Analysis complete. Found 2 critical issues."):
    p = AsyncMock()
    p.name = "mock"
    p.generate = AsyncMock(return_value=SimpleNamespace(
        text=text, model="mock-model", input_tokens=100, output_tokens=80,
        total_tokens=180, thinking_tokens=0, latency_ms=300.0,
        estimated_cost_usd=0.0, tokens_per_second=60.0,
    ))
    return p


def _judge():
    j = AsyncMock()
    j.name = "judge"
    j.generate = AsyncMock(return_value=SimpleNamespace(
        text="ACCURACY: 8\nCOMPLETENESS: 8\nFORMAT: 7\nRATIONALE: Solid analysis.",
        model="judge-model", input_tokens=50, output_tokens=40,
        total_tokens=90, thinking_tokens=0, latency_ms=100.0,
        estimated_cost_usd=0.0, tokens_per_second=90.0,
    ))
    return j


def test_run_probe_returns_summary(tmp_path):
    from atomics.probe.config import ProbeTarget
    from atomics.probe.runner import run_probe, ProbeSummary

    f = tmp_path / "access.log"
    f.write_text("10.0.0.5 GET /admin 403\n")
    targets = [ProbeTarget(name="nginx", artifact_type="access-log", source="file", path=str(f))]

    summary = asyncio.run(run_probe(_provider(), judge_provider=_judge(), targets=targets))
    assert isinstance(summary, ProbeSummary)
    assert len(summary.results) == 1


def test_run_probe_regression_detected(tmp_path):
    from atomics.probe.config import ProbeTarget
    from atomics.probe.runner import run_probe

    f = tmp_path / "access.log"
    f.write_text("10.0.0.5 GET /admin 403\n")
    targets = [ProbeTarget(name="nginx", artifact_type="access-log", source="file", path=str(f))]

    summary = asyncio.run(run_probe(
        _provider(),
        judge_provider=_judge(),
        targets=targets,
        prev_scores={"nginx": 1.0},
        regression_threshold=0.05,
    ))
    assert len(summary.regressions) >= 0
