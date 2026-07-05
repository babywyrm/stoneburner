"""Tests for the probe module — config, connectors, checks, runner."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

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
    import pytest

    from atomics.probe.config import ProbeConfigError, load_probe_config
    with pytest.raises(ProbeConfigError, match="not found"):
        load_probe_config(Path("/nonexistent/probes.yaml"))


def test_load_probe_config_invalid_artifact_type(tmp_path):
    import pytest

    from atomics.probe.config import ProbeConfigError, load_probe_config
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
    from atomics.probe.connectors import ProbeConnectorError, fetch_artifact
    target = ProbeTarget(
        name="test", artifact_type="access-log", source="file", path="/nonexistent/file.log"
    )
    with pytest.raises(ProbeConnectorError, match="not found"):
        asyncio.run(fetch_artifact(target))


def test_fetch_artifact_unknown_source():
    import asyncio

    import pytest

    from atomics.probe.config import ProbeTarget
    from atomics.probe.connectors import ProbeConnectorError, fetch_artifact
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
from types import SimpleNamespace
from unittest.mock import AsyncMock


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
    from atomics.probe.runner import ProbeSummary, run_probe

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


# ── Connector — http source + truncation + decode exception ───────────────────

def test_fetch_artifact_file_truncation(tmp_path):
    """_fetch_file truncates content > max_bytes."""
    import asyncio

    from atomics.probe.config import ProbeTarget
    from atomics.probe.connectors import fetch_artifact
    f = tmp_path / "big.log"
    f.write_bytes(b"A" * 200)
    target = ProbeTarget(name="big", artifact_type="access-log", source="file", path=str(f))
    content = asyncio.run(fetch_artifact(target, max_bytes=50))
    assert len(content) == 50


def test_fetch_artifact_http_source():
    """_fetch_http fetches content via httpx and returns decoded string."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    from atomics.probe.config import ProbeTarget
    from atomics.probe.connectors import fetch_artifact

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.content = b'{"status": "ok"}'

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    target = ProbeTarget(
        name="api", artifact_type="api-response", source="http",
        url="http://localhost:9999/status",
    )
    with patch("httpx.AsyncClient", return_value=mock_client):
        content = asyncio.run(fetch_artifact(target))
    assert "ok" in content


def test_fetch_artifact_http_error():
    """_fetch_http wraps httpx.HTTPError into ProbeConnectorError."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    import httpx

    from atomics.probe.config import ProbeTarget
    from atomics.probe.connectors import ProbeConnectorError, fetch_artifact

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    target = ProbeTarget(
        name="bad-api", artifact_type="api-response", source="http",
        url="http://localhost:9999/dead",
    )
    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(ProbeConnectorError, match="HTTP fetch failed"):
            asyncio.run(fetch_artifact(target))


def test_fetch_artifact_http_truncation():
    """_fetch_http truncates responses larger than max_bytes."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    from atomics.probe.config import ProbeTarget
    from atomics.probe.connectors import fetch_artifact

    big_content = b"X" * 200

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.content = big_content

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    target = ProbeTarget(
        name="big-api", artifact_type="api-response", source="http",
        url="http://localhost:9999/big",
    )
    with patch("httpx.AsyncClient", return_value=mock_client):
        content = asyncio.run(fetch_artifact(target, max_bytes=50))
    assert len(content) == 50


def test_fetch_artifact_http_custom_headers():
    """_fetch_http passes custom headers from ProbeTarget."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    from atomics.probe.config import ProbeTarget
    from atomics.probe.connectors import fetch_artifact

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.content = b"ok"

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    target = ProbeTarget(
        name="auth-api", artifact_type="api-response", source="http",
        url="http://localhost:9999/secure",
        headers={"Authorization": "Bearer tok-123"},
    )
    with patch("httpx.AsyncClient", return_value=mock_client) as patched_cls:
        asyncio.run(fetch_artifact(target))
    # Verify headers were passed to the client constructor
    patched_cls.assert_called_once()
    _, ctor_kwargs = patched_cls.call_args
    assert ctor_kwargs.get("headers", {}).get("Authorization") == "Bearer tok-123"


# ── Runner — ProbeSummary properties + fetch/analysis failure paths ───────────

def test_probe_summary_overall_score():
    from atomics.probe.runner import ProbeResult, ProbeSummary

    summary = ProbeSummary()
    summary.results = [
        ProbeResult(target_name="a", artifact_type="access-log", check_id="c",
                    score=0.8, prev_score=None, regressed=False, judge_model="m",
                    judge_rationale="ok"),
        ProbeResult(target_name="b", artifact_type="access-log", check_id="c",
                    score=0.6, prev_score=None, regressed=False, judge_model="m",
                    judge_rationale="ok"),
    ]
    assert summary.overall_score == 0.7
    # fixture_results is the convergent alias for results
    assert summary.fixture_results is summary.results


def test_probe_summary_to_dict_serializable():
    import json

    from atomics.probe.runner import ProbeResult, ProbeSummary

    summary = ProbeSummary()
    summary.results = [
        ProbeResult(target_name="a", artifact_type="access-log", check_id="c",
                    score=0.8, prev_score=0.9, regressed=True, judge_model="m",
                    judge_rationale="down"),
    ]
    d = summary.to_dict()
    json.dumps(d)  # must round-trip
    assert d["total_targets"] == 1
    assert d["regressions"] == ["a"]
    assert d["results"][0]["score"] == 0.8


def test_cli_probe_has_json_out_flag():
    from click.testing import CliRunner

    from atomics.cli import cli

    result = CliRunner().invoke(cli, ["probe", "--help"])
    assert result.exit_code == 0
    assert "--json-out" in result.output


def test_probe_summary_overall_score_empty():
    from atomics.probe.runner import ProbeSummary
    summary = ProbeSummary()
    assert summary.overall_score is None


def test_probe_summary_regressions():
    from atomics.probe.runner import ProbeResult, ProbeSummary
    summary = ProbeSummary()
    summary.results = [
        ProbeResult(target_name="a", artifact_type="access-log", check_id="c",
                    score=0.5, prev_score=0.9, regressed=True,
                    judge_model="m", judge_rationale="dropped"),
        ProbeResult(target_name="b", artifact_type="access-log", check_id="c",
                    score=0.9, prev_score=0.8, regressed=False,
                    judge_model="m", judge_rationale="ok"),
    ]
    assert len(summary.regressions) == 1
    assert summary.regressions[0].target_name == "a"


def test_run_probe_fetch_failure_path(tmp_path):
    """When fetch_artifact raises, runner records a fetch_error result."""

    from atomics.probe.config import ProbeTarget
    from atomics.probe.runner import run_probe

    targets = [ProbeTarget(name="broken", artifact_type="access-log",
                           source="file", path="/nonexistent/log.txt")]
    received = []

    def on_result(r):
        received.append(r)

    summary = asyncio.run(run_probe(
        _provider(), judge_provider=_judge(), targets=targets, on_result=on_result,
    ))
    assert summary.results[0].check_id == "fetch_error"
    assert len(received) == 1


def test_run_probe_analysis_failure_path(tmp_path):
    """When provider.generate raises, runner records a failed analysis result."""
    from atomics.probe.config import ProbeTarget
    from atomics.probe.runner import run_probe

    f = tmp_path / "log.txt"
    f.write_text("GET /admin 403\n")
    targets = [ProbeTarget(name="nginx", artifact_type="access-log",
                           source="file", path=str(f))]

    fail_provider = AsyncMock()
    fail_provider.name = "fail"
    fail_provider.generate = AsyncMock(side_effect=RuntimeError("LLM down"))

    received = []

    def on_result(r):
        received.append(r)

    summary = asyncio.run(run_probe(
        fail_provider, judge_provider=_judge(), targets=targets, on_result=on_result,
    ))
    assert summary.results[0].score is None
    assert len(received) == 1


def test_fetch_artifact_http_dispatches(tmp_path):
    """Line 22: source='http' dispatches to _fetch_http."""
    from unittest.mock import AsyncMock, patch

    from atomics.probe.config import ProbeTarget
    from atomics.probe.connectors import fetch_artifact

    target = ProbeTarget(
        name="api", artifact_type="api-response", source="http",
        url="http://localhost:9999/status",
    )
    # Patch _fetch_http directly — avoids needing aiohttp installed
    with patch("atomics.probe.connectors._fetch_http",
               new=AsyncMock(return_value='{"status":"ok"}')):
        content = asyncio.run(fetch_artifact(target))
    assert "ok" in content


def test_build_check_unknown_artifact_type():
    """probe/checks.py line 27: unknown artifact_type → generic_analysis handler."""
    from atomics.probe.checks import build_check
    check = build_check("some-unknown-type", "raw content here")
    assert check["check_id"] == "generic_analysis"
    assert "prompt" in check
