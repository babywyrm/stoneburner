"""In-memory REPL session. Never written to disk."""

from __future__ import annotations

from dataclasses import dataclass, field

SESSION_KEYS = ("provider", "model", "effort", "reasoning_mode", "host", "last_job_id")


class SessionError(ValueError):
    """Unknown session key or other local session mistake."""


@dataclass
class Session:
    provider: str | None = None
    model: str | None = None
    effort: str | None = None
    reasoning_mode: str | None = None
    host: str | None = None
    last_job_id: str | None = None
    _distributed_jobs: set[str] = field(default_factory=set, init=False, repr=False)

    def set(self, key: str, value: str | None) -> None:
        if key not in SESSION_KEYS:
            raise SessionError(f"unknown session key {key!r}; expected {', '.join(SESSION_KEYS)}")
        setattr(self, key, value)

    def remember_job(self, job_id: str, *, distributed: bool = False) -> None:
        self.last_job_id = job_id
        if distributed:
            self._distributed_jobs.add(job_id)

    def is_distributed_job(self, job_id: str) -> bool:
        return job_id in self._distributed_jobs

    def as_dict(self) -> dict[str, str | None]:
        return {key: getattr(self, key) for key in SESSION_KEYS}
