"""Compatibility shim. Implementation: ``atomics.load.scenario_models``."""

from __future__ import annotations

import sys

from atomics.load import scenario_models as _impl

sys.modules[__name__] = _impl
