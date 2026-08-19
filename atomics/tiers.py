"""Compatibility shim. Implementation: ``atomics.benchmark.tiers``."""

from __future__ import annotations

import sys

from atomics.benchmark import tiers as _impl

sys.modules[__name__] = _impl
