"""Compatibility shim. Implementation: ``atomics.benchmark.model_classes``."""

from __future__ import annotations

import sys

from atomics.benchmark import model_classes as _impl

sys.modules[__name__] = _impl
