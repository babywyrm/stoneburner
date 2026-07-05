"""LabCompare — compare two+ inference hosts on throughput and quality parity.

Additive module: imports existing providers/runners/judge as libraries and
never modifies them. Persists only to the labcompare_results table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("atomics.labcompare")


@dataclass(frozen=True)
class HostSpec:
    name: str
    url: str


def parse_host_specs(raw: list[str]) -> list[HostSpec]:
    """Parse ``NAME=URL`` strings into HostSpec objects.

    Raises ValueError on malformed input so the CLI can surface a clear error.
    """
    specs: list[HostSpec] = []
    for item in raw:
        if "=" not in item:
            raise ValueError(f"bad --host '{item}': expected NAME=URL")
        name, url = item.split("=", 1)
        name = name.strip()
        url = url.strip()
        if not name:
            raise ValueError(f"bad --host '{item}': empty host name")
        if not url:
            raise ValueError(f"bad --host '{item}': empty url")
        specs.append(HostSpec(name=name, url=url))
    return specs
