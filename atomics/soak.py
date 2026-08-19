"""Compatibility shim. Implementation: ``atomics.load.soak``."""

from __future__ import annotations

import sys

from atomics.load import soak as _impl

sys.modules[__name__] = _impl
