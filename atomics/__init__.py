"""Atomics — Agentic token usage benchmarking platform."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("atomics")
except PackageNotFoundError:
    # Imported from a source tree with no install (a bare `python -c` from the
    # repo root, for instance). The build backend reads pyproject directly, so
    # this only affects introspection, never packaging.
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
