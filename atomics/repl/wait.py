"""Poll get_job until completed, a poll budget, or Ctrl-C. Does not cancel."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from atomics.mcp.client import AtomicsApiClient

WAIT_INTERVAL_SECONDS = 2.0
WAIT_MAX_POLLS = 30


def _progress_sig(body: Any) -> Any:
    if not isinstance(body, dict):
        return None
    progress = body.get("progress") or {}
    in_flight = progress.get("in_flight")
    in_flight_sig: Any
    if isinstance(in_flight, dict):
        in_flight_sig = (
            in_flight.get("fixture_id"),
            in_flight.get("phase"),
            in_flight.get("model"),
        )
    else:
        in_flight_sig = in_flight
    return (progress.get("current"), in_flight_sig, body.get("status"))


def wait_for_job(
    client: AtomicsApiClient,
    job_id: str,
    *,
    sleep: Callable[[float], None],
    interval: float = WAIT_INTERVAL_SECONDS,
    max_polls: int = WAIT_MAX_POLLS,
    on_update: Callable[[Any], None] | None = None,
) -> Any:
    last: Any = None
    printed: Any = object()

    def emit(body: Any) -> None:
        nonlocal printed
        sig = _progress_sig(body)
        if on_update is not None and sig != printed:
            on_update(body)
            printed = sig

    for attempt in range(max_polls):
        last = client.get_job(job_id)
        emit(last)
        if isinstance(last, dict) and last.get("status") == "completed":
            return last
        if attempt + 1 >= max_polls:
            return last
        try:
            sleep(interval)
        except KeyboardInterrupt:
            return last
    return last
