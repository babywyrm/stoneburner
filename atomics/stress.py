"""Compatibility shim. Implementation: ``atomics.load.stress``."""

from __future__ import annotations

import sys

from atomics.load import stress as _impl

sys.modules[__name__] = _impl
