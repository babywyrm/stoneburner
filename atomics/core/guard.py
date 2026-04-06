"""Rate and budget guard — enforces token/request limits and cost caps."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass
class GuardConfig:
    max_tokens_per_hour: int = 100_000
    max_requests_per_minute: int = 30
    budget_limit_usd: float = 50.0
    circuit_breaker_threshold: int = 10


class RateBudgetGuard:
    """Tracks usage and decides whether the next request is allowed."""

    def __init__(self, config: GuardConfig) -> None:
        self._config = config
        self._request_timestamps: deque[float] = deque()
        self._hourly_tokens: deque[tuple[float, int]] = deque()
        self._total_cost: float = 0.0
        self._consecutive_errors: int = 0

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def circuit_open(self) -> bool:
        return self._consecutive_errors >= self._config.circuit_breaker_threshold

    def can_proceed(self) -> tuple[bool, str]:
        """Check all guards. Returns (allowed, reason)."""
        if self.circuit_open:
            return False, f"circuit breaker open ({self._consecutive_errors} consecutive errors)"

        if self._total_cost >= self._config.budget_limit_usd:
            return (
                False,
                f"budget exhausted (${self._total_cost:.2f} >= "
                f"${self._config.budget_limit_usd:.2f})",
            )

        now = time.monotonic()
        self._prune_timestamps(now)

        if len(self._request_timestamps) >= self._config.max_requests_per_minute:
            return False, f"rate limit ({self._config.max_requests_per_minute} req/min)"

        hour_tokens = sum(t for _, t in self._hourly_tokens)
        if hour_tokens >= self._config.max_tokens_per_hour:
            return False, f"hourly token cap ({hour_tokens}/{self._config.max_tokens_per_hour})"

        return True, "ok"

    def record_request(self, tokens: int, cost: float, success: bool) -> None:
        now = time.monotonic()
        self._request_timestamps.append(now)
        self._hourly_tokens.append((now, tokens))
        self._total_cost += cost

        if success:
            self._consecutive_errors = 0
        else:
            self._consecutive_errors += 1

    def reset_circuit(self) -> None:
        self._consecutive_errors = 0

    def seconds_until_allowed(self) -> float:
        """Estimate wait time if rate-limited on requests/min."""
        if not self._request_timestamps:
            return 0.0
        now = time.monotonic()
        self._prune_timestamps(now)
        if len(self._request_timestamps) < self._config.max_requests_per_minute:
            return 0.0
        oldest = self._request_timestamps[0]
        return max(0.0, 60.0 - (now - oldest))

    def _prune_timestamps(self, now: float) -> None:
        while self._request_timestamps and (now - self._request_timestamps[0]) > 60:
            self._request_timestamps.popleft()
        while self._hourly_tokens and (now - self._hourly_tokens[0][0]) > 3600:
            self._hourly_tokens.popleft()
