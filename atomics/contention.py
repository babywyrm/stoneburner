"""Compatibility shim. Implementation: ``atomics.load.contention``."""

from __future__ import annotations

import sys

from atomics.load import contention as _impl

sys.modules[__name__] = _impl
