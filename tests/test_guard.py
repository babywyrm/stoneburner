"""Tests for the rate/budget guard."""

import time

from atomics.core.guard import GuardConfig, RateBudgetGuard


def test_guard_allows_when_under_limits():
    guard = RateBudgetGuard(GuardConfig(max_requests_per_minute=10, max_tokens_per_hour=10000, budget_limit_usd=100.0))
    allowed, reason = guard.can_proceed()
    assert allowed is True
    assert reason == "ok"


def test_guard_blocks_on_budget():
    guard = RateBudgetGuard(GuardConfig(budget_limit_usd=1.0))
    guard.record_request(1000, 1.01, True)
    allowed, reason = guard.can_proceed()
    assert allowed is False
    assert "budget" in reason


def test_guard_blocks_on_request_rate():
    guard = RateBudgetGuard(GuardConfig(max_requests_per_minute=3, budget_limit_usd=100.0))
    for _ in range(3):
        guard.record_request(10, 0.001, True)
    allowed, reason = guard.can_proceed()
    assert allowed is False
    assert "rate limit" in reason


def test_guard_blocks_on_token_cap():
    guard = RateBudgetGuard(GuardConfig(max_tokens_per_hour=100, budget_limit_usd=100.0))
    guard.record_request(101, 0.001, True)
    allowed, reason = guard.can_proceed()
    assert allowed is False
    assert "token cap" in reason


def test_circuit_breaker_opens():
    guard = RateBudgetGuard(GuardConfig(circuit_breaker_threshold=3, budget_limit_usd=100.0))
    for _ in range(3):
        guard.record_request(0, 0.0, False)
    assert guard.circuit_open is True
    allowed, reason = guard.can_proceed()
    assert allowed is False
    assert "circuit breaker" in reason


def test_circuit_breaker_resets_on_success():
    guard = RateBudgetGuard(GuardConfig(circuit_breaker_threshold=3, budget_limit_usd=100.0))
    guard.record_request(0, 0.0, False)
    guard.record_request(0, 0.0, False)
    guard.record_request(10, 0.001, True)
    assert guard.circuit_open is False


def test_seconds_until_allowed():
    guard = RateBudgetGuard(GuardConfig(max_requests_per_minute=1, budget_limit_usd=100.0))
    guard.record_request(10, 0.001, True)
    wait = guard.seconds_until_allowed()
    assert wait > 0
    assert wait <= 60.0
