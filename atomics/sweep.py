"""Compatibility shim. Implementation: ``atomics.benchmark.sweep``."""

from __future__ import annotations

import sys

from atomics.benchmark import sweep as _impl

sys.modules[__name__] = _impl
