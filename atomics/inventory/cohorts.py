"""Group inventoried models into fair comparison sets for one battery."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any

from atomics.eval.batteries import Battery
from atomics.inventory import SCHEMA

# Inventory JSON as written by `atomics models --json-out`.
Record = dict[str, Any]

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


class CohortError(ValueError):
    pass


@dataclass
class Member:
    name: str
    hosts: tuple[str, ...]
    digest: str | None
    size_bytes: int
    band: str
    thinking: bool
    speed: dict[str, float] = field(default_factory=dict)
    host: str | None = None


@dataclass(frozen=True)
class Excluded:
    name: str
    reason: str


@dataclass
class Cohort:
    band: str
    members: list[Member]
    judge: Member | None = None


@dataclass
class Cohorts:
    battery: str
    needs_judge: bool
    taken_at: str | None
    cohorts: list[Cohort]
    excluded: list[Excluded]


def _has(record: Record, cap: str) -> bool:
    return (record.get("capabilities") or {}).get(cap, {}).get("value") is True


def _reason(record: Record, needs: frozenset[str], patterns: tuple[str, ...]) -> str | None:
    if patterns and not any(fnmatch(record["name"], p) for p in patterns):
        return "filtered by -m"
    if not record.get("evaluable"):
        return "not evaluable"
    missing = sorted(c for c in needs if not _has(record, c))
    return f"lacks {', '.join(missing)}" if missing else None


def _members(records: list[Record]) -> list[Member]:
    """One member per (tag, digest); tags with several digests are labelled tag@host."""
    by_key: dict[tuple[str, str | None], list[Record]] = {}
    for r in records:
        by_key.setdefault((r["name"], r.get("digest")), []).append(r)
    digests: dict[str, int] = {}
    for name, _ in by_key:
        digests[name] = digests.get(name, 0) + 1
    members = []
    for (name, digest), copies in by_key.items():
        first = copies[0]
        probed = [c for c in copies if c.get("probe")]
        probe = probed[0]["probe"] if probed else {}
        members.append(
            Member(
                name=f"{name}@{first['host']}" if digests[name] > 1 else name,
                hosts=tuple(c["host"] for c in copies),
                digest=digest,
                size_bytes=first.get("size_bytes") or 0,
                band=parameter_band(first.get("parameter_size"), first.get("model_class")) or "",
                thinking=probe.get("recommended") == "--thinking",
                speed={
                    c["host"]: (c["probe"].get("off") or {}).get("tokens_per_second") or 0.0
                    for c in probed
                },
            )
        )
    return members


def form_cohorts(
    inventory: Record,
    battery: Battery,
    patterns: tuple[str, ...] = (),
    max_members: int | None = None,
) -> Cohorts:
    """Group a saved inventory into cohorts for one battery."""
    if inventory.get("schema") != SCHEMA:
        raise CohortError(f"inventory schema {inventory.get('schema')!r}; expected {SCHEMA}")
    needs = battery_needs(battery)
    reasons: dict[str, str] = {}
    kept: list[Record] = []
    for record in inventory["models"]:
        reason = _reason(record, needs, patterns)
        if reason:
            reasons.setdefault(record["name"], reason)
        else:
            kept.append(record)
    excluded = {n: r for n, r in reasons.items() if n not in {k["name"] for k in kept}}
    members = _members(kept)
    excluded.update({m.name: "no size" for m in members if not m.band})
    cohorts = []
    for band in BANDS:
        group = sorted((m for m in members if m.band == band), key=lambda m: m.name)
        chunks = math.ceil(len(group) / max_members) if max_members else 1
        for i in range(chunks):
            chunk = group[i::chunks]
            if len(chunk) > 1:
                cohorts.append(Cohort(band, chunk))
            else:
                excluded.update({m.name: "alone in band" for m in chunk})
    return Cohorts(
        battery=battery.id,
        needs_judge=battery.needs_judge,
        taken_at=inventory.get("taken_at"),
        cohorts=cohorts,
        excluded=[Excluded(n, r) for n, r in excluded.items()],
    )
