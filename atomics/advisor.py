"""Compatibility shim. Implementation: ``atomics.benchmark.advisor``."""

from __future__ import annotations

import sys

from atomics.benchmark import advisor as _impl

sys.modules[__name__] = _impl
