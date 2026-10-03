"""Group inventoried models into fair comparison sets for one battery."""

from __future__ import annotations

import re

from atomics.eval.batteries import Battery

BANDS = ("<5B", "5-15B", "15-40B", ">40B")
_CLASS_BAND = {"light": "<5B", "mid": "5-15B", "heavy": "15-40B"}
_SUITE_NEEDS = {"toolcall": ("tools",)}
_PARAMS = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMBT])\s*$", re.IGNORECASE)
_BILLIONS = {"K": 1e-6, "M": 1e-3, "B": 1.0, "T": 1e3}


def parameter_band(parameter_size: str | None, model_class: str | None) -> str | None:
    """Band from the host's parameter count, else the name table's class."""
    match = _PARAMS.match(parameter_size or "")
    if match is None:
        return _CLASS_BAND.get(model_class or "")
    billions = float(match[1]) * _BILLIONS[match[2].upper()]
    if billions < 5:
        return "<5B"
    if billions < 15:
        return "5-15B"
    return "15-40B" if billions < 40 else ">40B"


def battery_needs(battery: Battery) -> frozenset[str]:
    """Capabilities every model needs to run the battery's required steps."""
    return frozenset({"completion"}).union(*(_SUITE_NEEDS.get(s.suite, ()) for s in battery.steps))
