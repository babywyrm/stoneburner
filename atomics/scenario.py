"""Compatibility shim. Implementation: ``atomics.load.scenario``."""

from __future__ import annotations

import sys

from atomics.load import scenario as _impl

sys.modules[__name__] = _impl
