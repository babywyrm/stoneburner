"""Server configuration for atomics API mode."""

from __future__ import annotations

import ipaddress
import math
from dataclasses import dataclass, field
from pathlib import Path

from atomics.paths import default_db_path
from atomics.validation import AllowedEndpoint, is_endpoint_allowed, parse_allowed_endpoint


class DisallowedHostError(PermissionError):
    """Raised when a caller-supplied outbound endpoint is not in `ServerSettings.allowed_hosts`."""


def is_loopback_host(host: str) -> bool:
    """Return True if `host` can only be reached from this machine."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


@dataclass
class ServerSettings:
    """Settings for the atomics API server."""

    host: str = "127.0.0.1"
    port: int = 8000
    api_keys: set[str] = field(default_factory=set)
    # Keys accepted on worker lifecycle endpoints only. Left empty, workers fall
    # back to `api_keys`, which means a worker credential also authorizes run and
    # eval submission. Set this whenever workers run on hosts you do not control.
    worker_api_keys: set[str] = field(default_factory=set)
    no_auth: bool = False
    log_level: str = "info"
    db_path: Path = field(default_factory=default_db_path)
    # Roughly four missed heartbeats at the worker's default 30s interval. It has
    # to be settable because that interval is: `atomics worker
    # --heartbeat-interval 300` against a fixed 120s threshold would have every
    # worker declared absent, and its pinned fleet work failed, while behaving
    # exactly as configured.
    worker_absent_after_seconds: float = 120.0
    with_dashboard: bool = False
    # Job state is in-process, so both of these are memory bounds. Active caps
    # how much work runs at once; retained caps how many finished results stay
    # pollable afterwards.
    max_active_jobs: int = 16
    max_retained_jobs: int = 256
    # One caller's share of the active budget. Without it the global limit is
    # first-come-first-served, so one impatient script starves every other key.
    max_active_jobs_per_caller: int = 4
    # Optional cumulative dollar ceiling per caller across runs in this server
    # process. None leaves per-run budgets as the only spend bound.
    max_spend_usd_per_caller: float | None = None
    # Optional allowlist of outbound HTTP(S) origins or URL prefixes callers may
    # target via `host`, `judge_host`, battery profiles, or distributed runs.
    allowed_hosts: set[str] | None = None
    _compiled_allowed_hosts: tuple[AllowedEndpoint, ...] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"port must be in range 1-65535, got {self.port}")
        if self.worker_absent_after_seconds <= 0:
            raise ValueError(
                "worker_absent_after_seconds must be positive, got "
                f"{self.worker_absent_after_seconds}"
            )
        if self.max_active_jobs < 1:
            raise ValueError(f"max_active_jobs must be positive, got {self.max_active_jobs}")
        if self.max_retained_jobs < 1:
            raise ValueError(f"max_retained_jobs must be positive, got {self.max_retained_jobs}")
        if self.max_active_jobs_per_caller < 1:
            raise ValueError(
                "max_active_jobs_per_caller must be positive, got "
                f"{self.max_active_jobs_per_caller}"
            )
        if self.max_spend_usd_per_caller is not None and (
            isinstance(self.max_spend_usd_per_caller, bool)
            or not math.isfinite(self.max_spend_usd_per_caller)
            or self.max_spend_usd_per_caller <= 0
        ):
            raise ValueError(
                f"max_spend_usd_per_caller must be positive, got {self.max_spend_usd_per_caller}"
            )
        if self.allowed_hosts is not None:
            if not self.allowed_hosts:
                raise ValueError(
                    "allowed_hosts must not be empty when set; omit it to allow any valid endpoint"
                )
            self._compiled_allowed_hosts = tuple(
                parse_allowed_endpoint(item, label="allowed_hosts")
                for item in sorted(self.allowed_hosts)
            )
        if self.no_auth and not is_loopback_host(self.host):
            raise ValueError(
                f"no_auth cannot be combined with the non-loopback host {self.host!r}: "
                "that exposes every endpoint, including eval submission, to the "
                "network unauthenticated. Bind to 127.0.0.1 or supply an API key."
            )

    def check_outbound_host(self, url: str | None, *, label: str = "host") -> None:
        """Validate a caller-supplied outbound endpoint against `allowed_hosts` when configured."""
        if url is None or self._compiled_allowed_hosts is None:
            return
        if not is_endpoint_allowed(url, self._compiled_allowed_hosts, label=label):
            raise DisallowedHostError(
                f"{label} is not permitted by the server's outbound host allowlist"
            )

    @property
    def effective_worker_keys(self) -> set[str]:
        """Keys accepted on worker endpoints, falling back to the submitter keys."""
        return self.worker_api_keys or self.api_keys
