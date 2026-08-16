"""Atomics — Agentic token usage benchmarking platform."""

from importlib.metadata import PackageNotFoundError, version

# PyPI / uv listing name. The importable package and CLI stay `atomics`.
DIST_NAME = "stoneburner-atomics"

try:
    __version__ = version(DIST_NAME)
except PackageNotFoundError:
    # Imported from a source tree with no install (a bare `python -c` from the
    # repo root, for instance). The build backend reads pyproject directly, so
    # this only affects introspection, never packaging.
    __version__ = "0.0.0+unknown"

__all__ = ["DIST_NAME", "__version__"]
