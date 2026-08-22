"""In-memory REPL session. Never written to disk."""

from __future__ import annotations

from dataclasses import asdict, dataclass

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

    def set(self, key: str, value: str | None) -> None:
        if key not in SESSION_KEYS:
            raise SessionError(
                f"unknown session key {key!r}; expected {', '.join(SESSION_KEYS)}"
            )
        setattr(self, key, value)

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)
