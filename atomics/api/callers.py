"""Caller identity for per-key accounting and request logs.

A caller is identified by a short digest of the API key they presented, never
by the key itself. The identifier is stable across requests, so quotas and log
lines can be attributed to one caller, but it cannot be turned back into a
credential — which matters because it ends up in log files that are read,
shipped, and retained far more casually than secrets are.
"""

from __future__ import annotations

import hashlib

# Used when no credential distinguishes one caller from another: `--no-auth`
# mode, or an unauthenticated endpoint. Per-caller quotas cannot partition
# anything in that case, and that is a property of running without auth rather
# than something this module can fix.
ANONYMOUS_CALLER = "anonymous"

# Enough to keep collisions implausible among the handful of keys a deployment
# configures, short enough to stay readable in a log line.
_DIGEST_CHARS = 12


def caller_id_from_key(key: str) -> str:
    """Derive a stable, non-reversible identifier for an API key."""
    if not key:
        return ANONYMOUS_CALLER
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]
