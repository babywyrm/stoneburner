"""Compatibility shim. Implementation: ``atomics.load.scenario_prompts``."""

from __future__ import annotations

import sys

from atomics.load import scenario_prompts as _impl

sys.modules[__name__] = _impl
