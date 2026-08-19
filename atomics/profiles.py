"""Compatibility shim. Implementation: ``atomics.load.profiles``."""

from __future__ import annotations

import sys

from atomics.load import profiles as _impl

sys.modules[__name__] = _impl
