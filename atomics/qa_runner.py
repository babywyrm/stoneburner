"""Compatibility shim. Implementation: ``atomics.benchmark.qa_runner``."""

from __future__ import annotations

import sys

from atomics.benchmark import qa_runner as _impl

sys.modules[__name__] = _impl
