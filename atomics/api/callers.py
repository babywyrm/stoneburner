"""Caller identity for per-key accounting and request logs.

A caller is identified by a short digest of the API key they presented, never
by the key itself. The identifier is stable across requests, so quotas and log
lines can be attributed to one caller, but it cannot be turned back into a
credential — which matters because it ends up in log files that are read,
shipped, and retained far more casually than secrets are.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import deque
from collections.abc import Callable

# Used when no credential distinguishes one caller from another: `--no-auth`
# mode, or an unauthenticated endpoint. Per-caller quotas cannot partition
# anything in that case, and that is a property of running without auth rather
# than something this module can fix.
ANONYMOUS_CALLER = "anonymous"

# Enough to keep collisions implausible among the handful of keys a deployment
# configures, short enough to stay readable in a log line.
_DIGEST_CHARS = 12
_RATE_WINDOW_SECONDS = 60.0


def caller_id_from_key(key: str) -> str:
    """Derive a stable, non-reversible identifier for an API key."""
    if not key:
        return ANONYMOUS_CALLER
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]


class CallerRateLimitExceededError(RuntimeError):
    """Raised when an authenticated caller exceeds their requests-per-minute quota."""

    def __init__(self, *, limit: int, retry_after: int) -> None:
        super().__init__(
            f"caller exceeded request rate limit ({limit}/min); retry in {retry_after}s"
        )
        self.limit = limit
        self.retry_after = retry_after


class CallerRateLimiter:
    """In-memory 60-second sliding-window request rate limiter keyed by caller digest."""

    def __init__(
        self,
        max_requests_per_minute: int | None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_requests_per_minute = max_requests_per_minute
        self._clock = clock
        self._buckets: dict[str, deque[float]] = {}

    def check(self, caller: str) -> None:
        """Record one request for `caller` or raise `CallerRateLimitExceededError`."""
        limit = self.max_requests_per_minute
        if limit is None:
            return
        now = self._clock()
        cutoff = now - _RATE_WINDOW_SECONDS
        bucket = self._buckets.get(caller)
        if bucket is not None:
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if not bucket:
                self._buckets.pop(caller, None)
                bucket = None
        if bucket is not None and len(bucket) >= limit:
            retry_after = max(1, math.ceil(_RATE_WINDOW_SECONDS - (now - bucket[0])))
            raise CallerRateLimitExceededError(limit=limit, retry_after=retry_after)
        if bucket is None:
            bucket = deque()
            self._buckets[caller] = bucket
        bucket.append(now)
