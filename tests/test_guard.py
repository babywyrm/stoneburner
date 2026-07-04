"""Tests for the rate/budget guard."""

from atomics.core.guard import GuardConfig, RateBudgetGuard


def test_guard_allows_when_under_limits():
    guard = RateBudgetGuard(
        GuardConfig(
            max_requests_per_minute=10,
            max_tokens_per_hour=10000,
            budget_limit_usd=100.0,
        )
    )
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


# ── Missing guard branches ────────────────────────────────────────────────────

def test_guard_total_cost_property():
    from atomics.core.guard import GuardConfig, RateBudgetGuard
    g = RateBudgetGuard(GuardConfig(budget_limit_usd=5.0))
    assert g.total_cost == 0.0
    g.record_request(100, 1.25, True)
    assert g.total_cost == 1.25


def test_guard_reset_circuit():
    from atomics.core.guard import GuardConfig, RateBudgetGuard
    g = RateBudgetGuard(GuardConfig(circuit_breaker_threshold=2))
    g.record_request(0, 0.0, False)
    g.record_request(0, 0.0, False)
    assert g.circuit_open
    g.reset_circuit()
    assert not g.circuit_open


def test_guard_seconds_until_allowed_empty():
    from atomics.core.guard import GuardConfig, RateBudgetGuard
    g = RateBudgetGuard(GuardConfig(max_requests_per_minute=60))
    assert g.seconds_until_allowed() == 0.0


def test_guard_seconds_until_allowed_below_limit():
    from atomics.core.guard import GuardConfig, RateBudgetGuard
    g = RateBudgetGuard(GuardConfig(max_requests_per_minute=60))
    g.record_request(10, 0.0, True)
    # one request, limit is 60 → not rate limited → 0.0
    assert g.seconds_until_allowed() == 0.0


def test_guard_seconds_until_allowed_at_limit():
    from atomics.core.guard import GuardConfig, RateBudgetGuard
    g = RateBudgetGuard(GuardConfig(max_requests_per_minute=1))
    g.record_request(10, 0.001, True)
    wait = g.seconds_until_allowed()
    assert 0.0 < wait <= 60.0


def test_guard_prune_old_timestamps():
    """Lines 87, 89: _prune_timestamps removes entries older than 60s / 3600s."""
    import time

    from atomics.core.guard import GuardConfig, RateBudgetGuard
    g = RateBudgetGuard(GuardConfig(max_requests_per_minute=100))
    # Manually inject a timestamp that is 120 seconds old
    g._request_timestamps.appendleft(time.monotonic() - 120)
    g._hourly_tokens.appendleft((time.monotonic() - 4000, 50))
    old_req_len = len(g._request_timestamps)
    old_tok_len = len(g._hourly_tokens)
    # Calling can_proceed triggers _prune_timestamps
    g.can_proceed()
    assert len(g._request_timestamps) < old_req_len
    assert len(g._hourly_tokens) < old_tok_len
