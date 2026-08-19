"""Compatibility shim. Implementation: ``atomics.benchmark.labcompare``."""

from __future__ import annotations

import sys

from atomics.benchmark import labcompare as _impl

sys.modules[__name__] = _impl
