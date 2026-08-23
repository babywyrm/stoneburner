"""Poll get_job until the job is done or Ctrl-C. Does not cancel."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from atomics.mcp.client import AtomicsApiClient

WAIT_INTERVAL_SECONDS = 2.0
WAIT_MAX_POLLS: int | None = None
_DONE = frozenset({"completed", "failed"})


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
            in_flight.get("suite"),
            in_flight.get("concurrency"),
            in_flight.get("elapsed_seconds"),
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
    max_polls: int | None = WAIT_MAX_POLLS,
    on_update: Callable[[Any], None] | None = None,
) -> Any:
    last: Any = None
    printed: Any = object()
    attempt = 0

    def emit(body: Any) -> None:
        nonlocal printed
        sig = _progress_sig(body)
        if on_update is not None and sig != printed:
            on_update(body)
            printed = sig

    while True:
        last = client.get_job(job_id)
        emit(last)
        if isinstance(last, dict) and last.get("status") in _DONE:
            return last
        attempt += 1
        if max_polls is not None and attempt >= max_polls:
            return last
        try:
            sleep(interval)
        except KeyboardInterrupt:
            return last
