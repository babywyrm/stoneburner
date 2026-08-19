"""Compatibility shim. Implementation: ``atomics.load.regression``."""

from __future__ import annotations

import sys

from atomics.load import regression as _impl

sys.modules[__name__] = _impl
