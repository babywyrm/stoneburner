"""Tests for webhook notifications."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from atomics.models import RunSummary
from atomics.webhooks import (
    _build_discord_payload,
    _build_generic_payload,
    _build_slack_payload,
    _is_discord_url,
    _is_slack_url,
    check_regression,
    send_webhook,
)


def _make_summary(
    *,
    total_tasks: int = 10,
    successful: int = 10,
    failed: int = 0,
    tokens: int = 1000,
    cost: float = 0.05,
    latency: float = 500.0,
) -> RunSummary:
    return RunSummary(
        run_id="test-run-123",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        total_tasks=total_tasks,
        successful_tasks=successful,
        failed_tasks=failed,
        total_tokens=tokens,
        total_cost_usd=cost,
        avg_latency_ms=latency,
    )


# ── URL detection ────────────────────────────────────────────────────────────


def test_is_slack_url():
    assert _is_slack_url("https://hooks.slack.com/services/T00/B00/xxx") is True
    assert _is_slack_url("https://example.com/webhook") is False


def test_is_discord_url():
    assert _is_discord_url("https://discord.com/api/webhooks/123/abc") is True
    assert _is_discord_url("https://example.com/webhook") is False


# ── Payload builders ─────────────────────────────────────────────────────────


def test_slack_payload_structure():
    summary = _make_summary()
    payload = _build_slack_payload(summary, tier="ez", provider="bedrock")
    assert "blocks" in payload
    assert len(payload["blocks"]) == 2
    assert payload["blocks"][0]["type"] == "header"


def test_slack_payload_with_alert():
    summary = _make_summary(failed=3)
    payload = _build_slack_payload(summary, tier="ez", provider="claude", alert="Latency spike")
    header_text = payload["blocks"][0]["text"]["text"]
    assert "Latency spike" in header_text


def test_discord_payload_structure():
    summary = _make_summary()
    payload = _build_discord_payload(summary, tier="baseline", provider="openai")
    assert "embeds" in payload
    assert payload["embeds"][0]["color"] == 0x2ECC71


def test_discord_payload_failure_color():
    summary = _make_summary(failed=2)
    payload = _build_discord_payload(summary, tier="ez", provider="claude")
    assert payload["embeds"][0]["color"] == 0xE74C3C


def test_generic_payload_structure():
    summary = _make_summary()
    payload = _build_generic_payload(summary, tier="mega", provider="groq")
    assert payload["event"] == "run_complete"
    assert payload["provider"] == "groq"
    assert payload["tier"] == "mega"
    assert payload["total_tasks"] == 10
    assert payload["alert"] is None


def test_generic_payload_with_alert():
    summary = _make_summary()
    payload = _build_generic_payload(summary, tier="ez", provider="claude", alert="Regression!")
    assert payload["event"] == "regression"
    assert payload["alert"] == "Regression!"


# ── Regression detection ─────────────────────────────────────────────────────


def test_no_regression():
    summary = _make_summary(latency=500.0)
    result = check_regression(summary, previous_avg_latency=480.0, previous_success_rate=100.0)
    assert result is None


def test_latency_regression():
    summary = _make_summary(latency=800.0)
    result = check_regression(summary, previous_avg_latency=500.0)
    assert result is not None
    assert "Latency" in result


def test_success_rate_regression():
    summary = _make_summary(total_tasks=10, successful=7, failed=3)
    result = check_regression(summary, previous_success_rate=100.0)
    assert result is not None
    assert "Success rate" in result


def test_high_failure_rate():
    summary = _make_summary(total_tasks=10, successful=5, failed=5)
    result = check_regression(summary)
    assert result is not None
    assert "failure rate" in result.lower()


def test_no_regression_within_threshold():
    summary = _make_summary(latency=520.0)
    result = check_regression(summary, previous_avg_latency=500.0)
    assert result is None


# ── Send webhook ─────────────────────────────────────────────────────────────


class FakeHttpClient(httpx.Client):
    def __init__(self, status_code: int = 200):
        super().__init__()
        self._status_code = status_code
        self.calls: list[dict] = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        request = httpx.Request("POST", url)
        return httpx.Response(self._status_code, request=request)

    def close(self):
        pass


def test_send_webhook_success():
    client = FakeHttpClient(200)
    summary = _make_summary()
    result = send_webhook(
        "https://example.com/webhook",
        summary, tier="ez", provider="claude",
        client=client,
    )
    assert result is True
    assert len(client.calls) == 1


def test_send_webhook_failure():
    client = FakeHttpClient(500)
    summary = _make_summary()
    result = send_webhook(
        "https://example.com/webhook",
        summary, tier="ez", provider="claude",
        client=client,
    )
    assert result is False


def test_send_webhook_slack_format():
    client = FakeHttpClient(200)
    summary = _make_summary()
    send_webhook(
        "https://hooks.slack.com/services/T00/B00/xxx",
        summary, tier="ez", provider="claude",
        client=client,
    )
    payload = client.calls[0]["json"]
    assert "blocks" in payload


def test_send_webhook_discord_format():
    client = FakeHttpClient(200)
    summary = _make_summary()
    send_webhook(
        "https://discord.com/api/webhooks/123/abc",
        summary, tier="ez", provider="claude",
        client=client,
    )
    payload = client.calls[0]["json"]
    assert "embeds" in payload


def test_send_webhook_generic_format():
    client = FakeHttpClient(200)
    summary = _make_summary()
    send_webhook(
        "https://example.com/my-hook",
        summary, tier="ez", provider="claude",
        client=client,
    )
    payload = client.calls[0]["json"]
    assert "event" in payload
    assert payload["event"] == "run_complete"
