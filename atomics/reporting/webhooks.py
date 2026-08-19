"""Webhook notifications — Slack, Discord, and generic HTTP POST.

Sends structured JSON payloads on run completion or regression detection.
Configure via ATOMICS_WEBHOOK_URL environment variable or --webhook flag.

Supports:
- Slack incoming webhooks (auto-detected by URL)
- Discord webhooks (auto-detected by URL)
- Generic HTTP POST (JSON body)
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from atomics.models import RunSummary

logger = logging.getLogger("atomics.webhooks")


def _is_slack_url(url: str) -> bool:
    return "hooks.slack.com" in url


def _is_discord_url(url: str) -> bool:
    return "discord.com/api/webhooks" in url


def _build_slack_payload(
    summary: RunSummary,
    *,
    tier: str,
    provider: str,
    alert: str | None = None,
) -> dict[str, Any]:
    status = ":white_check_mark:" if summary.failed_tasks == 0 else ":warning:"
    header = f"{status} Atomics Run Complete"
    if alert:
        header = f":rotating_light: {alert}"

    return {
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": header},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Run ID:* `{summary.run_id}`"},
                    {"type": "mrkdwn", "text": f"*Provider:* {provider}"},
                    {"type": "mrkdwn", "text": f"*Tier:* {tier}"},
                    {"type": "mrkdwn", "text": f"*Tasks:* {summary.successful_tasks}/{summary.total_tasks} OK"},
                    {"type": "mrkdwn", "text": f"*Tokens:* {summary.total_tokens:,}"},
                    {"type": "mrkdwn", "text": f"*Cost:* ${summary.total_cost_usd:.4f}"},
                ],
            },
        ],
    }


def _build_discord_payload(
    summary: RunSummary,
    *,
    tier: str,
    provider: str,
    alert: str | None = None,
) -> dict[str, Any]:
    status = "\u2705" if summary.failed_tasks == 0 else "\u26a0\ufe0f"
    title = f"{status} Atomics Run Complete"
    if alert:
        title = f"\U0001f6a8 {alert}"

    return {
        "embeds": [
            {
                "title": title,
                "color": 0x2ECC71 if summary.failed_tasks == 0 else 0xE74C3C,
                "fields": [
                    {"name": "Run ID", "value": f"`{summary.run_id}`", "inline": True},
                    {"name": "Provider", "value": provider, "inline": True},
                    {"name": "Tier", "value": tier, "inline": True},
                    {"name": "Tasks", "value": f"{summary.successful_tasks}/{summary.total_tasks}", "inline": True},
                    {"name": "Tokens", "value": f"{summary.total_tokens:,}", "inline": True},
                    {"name": "Cost", "value": f"${summary.total_cost_usd:.4f}", "inline": True},
                ],
            }
        ],
    }


def _build_generic_payload(
    summary: RunSummary,
    *,
    tier: str,
    provider: str,
    alert: str | None = None,
) -> dict[str, Any]:
    return {
        "event": "run_complete" if not alert else "regression",
        "alert": alert,
        "run_id": summary.run_id,
        "provider": provider,
        "tier": tier,
        "total_tasks": summary.total_tasks,
        "successful_tasks": summary.successful_tasks,
        "failed_tasks": summary.failed_tasks,
        "total_tokens": summary.total_tokens,
        "total_cost_usd": round(summary.total_cost_usd, 6),
        "avg_latency_ms": round(summary.avg_latency_ms, 1),
    }


def send_webhook(
    url: str,
    summary: RunSummary,
    *,
    tier: str = "baseline",
    provider: str = "unknown",
    alert: str | None = None,
    timeout: float = 10.0,
    client: httpx.Client | None = None,
) -> bool:
    """Send a webhook notification. Returns True on success, False on failure."""
    if _is_slack_url(url):
        payload = _build_slack_payload(summary, tier=tier, provider=provider, alert=alert)
    elif _is_discord_url(url):
        payload = _build_discord_payload(summary, tier=tier, provider=provider, alert=alert)
    else:
        payload = _build_generic_payload(summary, tier=tier, provider=provider, alert=alert)

    http = client or httpx.Client()
    try:
        resp = http.post(url, json=payload, timeout=timeout)
        if resp.status_code < 300:
            logger.info("Webhook sent to %s (status %d)", url, resp.status_code)
            return True
        logger.warning("Webhook failed: %s returned %d", url, resp.status_code)
        return False
    except Exception:
        logger.warning("Webhook delivery failed", exc_info=True)
        return False
    finally:
        if client is None:
            http.close()


def check_regression(
    current: RunSummary,
    previous_avg_latency: float | None = None,
    previous_success_rate: float | None = None,
    *,
    latency_threshold_pct: float = 25.0,
    success_threshold_pct: float = 10.0,
) -> str | None:
    """Check if the current run shows regression vs previous metrics.

    Returns an alert message if regression detected, None otherwise.
    """
    alerts: list[str] = []

    if previous_avg_latency is not None and current.avg_latency_ms > 0:
        increase_pct = (
            (current.avg_latency_ms - previous_avg_latency) / previous_avg_latency * 100
        )
        if increase_pct > latency_threshold_pct:
            alerts.append(f"Latency +{increase_pct:.0f}% ({previous_avg_latency:.0f}ms -> {current.avg_latency_ms:.0f}ms)")

    if previous_success_rate is not None and current.total_tasks > 0:
        current_rate = current.successful_tasks / current.total_tasks * 100
        drop = previous_success_rate - current_rate
        if drop > success_threshold_pct:
            alerts.append(f"Success rate -{drop:.0f}% ({previous_success_rate:.0f}% -> {current_rate:.0f}%)")

    if current.total_tasks > 0 and current.failed_tasks / current.total_tasks > 0.2:
        alerts.append(f"High failure rate: {current.failed_tasks}/{current.total_tasks}")

    return " | ".join(alerts) if alerts else None
