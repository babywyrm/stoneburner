"""Reporting package. ``from atomics.reporting import …`` still resolves."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_SUBMODULES = frozenset({"exporters", "hooks", "reporting", "stats", "webhooks"})


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return import_module(f"atomics.reporting.{name}")
    return getattr(import_module("atomics.reporting.reporting"), name)


def __dir__() -> list[str]:
    reporting = import_module("atomics.reporting.reporting")
    return sorted(set(_SUBMODULES) | set(dir(reporting)))
