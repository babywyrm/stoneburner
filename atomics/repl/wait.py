"""Poll get_job until the job is done or Ctrl-C. Does not cancel."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from atomics.mcp.client import AtomicsApiClient, AtomicsApiError

WAIT_INTERVAL_SECONDS = 2.0
WAIT_MAX_POLLS: int | None = None
_DONE = frozenset({"completed", "partial", "failed"})


def _progress_sig(body: Any) -> Any:
    if not isinstance(body, dict):
        return None
    if "mode" in body and "request_json" in body:
        return ("distributed", body.get("status"), body.get("summary_json"))
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
            in_flight.get("task"),
        )
    else:
        in_flight_sig = in_flight
    trail = progress.get("trail") or []
    trail_len = len(trail) if isinstance(trail, list) else 0
    return (progress.get("current"), in_flight_sig, body.get("status"), trail_len)


def wait_for_job(
    client: AtomicsApiClient,
    job_id: str,
    *,
    sleep: Callable[[float], None],
    interval: float = WAIT_INTERVAL_SECONDS,
    max_polls: int | None = WAIT_MAX_POLLS,
    on_update: Callable[[Any], None] | None = None,
    distributed: bool = False,
) -> Any:
    last: Any = None
    printed: Any = object()
    attempt = 0
    use_distributed = distributed

    def emit(body: Any) -> None:
        nonlocal printed
        sig = _progress_sig(body)
        if on_update is not None and sig != printed:
            on_update(body)
            printed = sig

    while True:
        if use_distributed:
            last = client.get_distributed_run(job_id)
        else:
            try:
                last = client.get_job(job_id)
            except AtomicsApiError as exc:
                if attempt == 0 and exc.status_code == 404:
                    try:
                        last = client.get_distributed_run(job_id)
                    except AtomicsApiError:
                        raise exc from None
                    use_distributed = True
                else:
                    raise
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
