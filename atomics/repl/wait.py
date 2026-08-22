"""Poll get_job until completed, a poll budget, or Ctrl-C. Does not cancel."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from atomics.mcp.client import AtomicsApiClient

WAIT_INTERVAL_SECONDS = 2.0
WAIT_MAX_POLLS = 30


def wait_for_job(
    client: AtomicsApiClient,
    job_id: str,
    *,
    sleep: Callable[[float], None],
    interval: float = WAIT_INTERVAL_SECONDS,
    max_polls: int = WAIT_MAX_POLLS,
) -> Any:
    last: Any = None
    for attempt in range(max_polls):
        last = client.get_job(job_id)
        if isinstance(last, dict) and last.get("status") == "completed":
            return last
        if attempt + 1 >= max_polls:
            return last
        try:
            sleep(interval)
        except KeyboardInterrupt:
            return last
    return last
