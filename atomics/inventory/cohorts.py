"""Group inventoried models into fair comparison sets for one battery."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any

from atomics.eval.batteries import Battery
from atomics.inventory import JUDGE_FIT_MARGIN, SCHEMA

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
    hosts: list[Record] = field(default_factory=list)


@dataclass(frozen=True)
class PlanJob:
    battery: str
    model: str
    host: str
    provider: str
    thinking: bool
    judge: str | None = None
    judge_host: str | None = None
    band: str = ""


def _host_row(hosts: list[Record], label: str | None) -> Record:
    return next((h for h in hosts if h.get("label") == label), {})


def _resolve(hosts: list[Record], label: str | None) -> tuple[str, str]:
    row = _host_row(hosts, label)
    return str(row.get("url") or label or ""), str(row.get("provider") or "ollama")


def plan_jobs(result: Cohorts) -> list[PlanJob]:
    """One battery-run job per placed member."""
    jobs: list[PlanJob] = []
    for cohort in result.cohorts:
        judge = _tag(cohort.judge.name) if cohort.judge else None
        judge_host = _resolve(result.hosts, cohort.judge.host)[0] if cohort.judge else None
        for member in cohort.members:
            host, provider = _resolve(result.hosts, member.host)
            jobs.append(
                PlanJob(
                    battery=result.battery,
                    model=_tag(member.name),
                    host=host,
                    provider=provider,
                    thinking=member.thinking,
                    judge=judge,
                    judge_host=judge_host,
                    band=cohort.band,
                )
            )
    return jobs


def job_argv(job: PlanJob) -> list[str]:
    args = ["battery", "run", job.battery, "-m", job.model, "-p", job.provider]
    args.append("--thinking" if job.thinking else "--no-thinking")
    if job.provider in {"vllm", "llamacpp"}:
        args.extend(["--vllm-host", job.host])
    else:
        args.extend(["--ollama-host", job.host])
    if job.judge:
        args.extend(["--judge-model", job.judge])
        if job.judge_host:
            args.extend(["--judge-host", job.judge_host])
    return args


def jobs_from_plan(data: Record) -> list[PlanJob]:
    if data.get("schema") != SCHEMA:
        raise CohortError(f"cohorts schema {data.get('schema')!r}; expected {SCHEMA}")
    raw = data.get("jobs")
    if not isinstance(raw, list):
        raise CohortError("no jobs; re-run atomics cohorts --json-out")
    battery = str(data.get("battery") or "")
    jobs: list[PlanJob] = []
    for row in raw:
        judge = row.get("judge") if isinstance(row, dict) else None
        name = host = None
        if isinstance(judge, dict):
            name, host = judge.get("name"), judge.get("host")
        jobs.append(
            PlanJob(
                battery=battery,
                model=str(row["model"]),
                host=str(row["host"]),
                provider=str(row.get("provider") or "ollama"),
                thinking=bool(row.get("thinking")),
                judge=name,
                judge_host=host,
            )
        )
    return jobs


def to_dict(result: Cohorts) -> dict[str, object]:
    def where(m: Member) -> dict[str, object]:
        return {"name": m.name, "host": m.host}

    return {
        "schema": SCHEMA,
        "inventory_taken_at": result.taken_at,
        "battery": result.battery,
        "jobs": [
            {
                "model": job.model,
                "host": job.host,
                "provider": job.provider,
                "thinking": job.thinking,
                "judge": ({"name": job.judge, "host": job.judge_host} if job.judge else None),
                "band": job.band,
            }
            for job in plan_jobs(result)
        ],
        "cohorts": [
            {
                "band": c.band,
                "judge": where(c.judge) if c.judge else None,
                "members": [
                    {**where(m), "digest": m.digest, "thinking": m.thinking} for m in c.members
                ],
            }
            for c in result.cohorts
        ],
        "excluded": [{"name": x.name, "reason": x.reason} for x in result.excluded],
    }


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


def _place(members: list[Member], load: dict[str, int], order: list[str]) -> None:
    """Greedy: biggest first, each to its least-loaded host; faster host breaks ties."""
    for m in sorted(members, key=lambda m: -m.size_bytes):
        m.host = min(
            m.hosts,
            key=lambda h: (
                load.get(h, 0),
                -m.speed.get(h, 0.0),
                order.index(h) if h in order else len(order),
            ),
        )
        load[m.host] = load.get(m.host, 0) + m.size_bytes


def _margin(record: Record) -> float | None:
    judge = record.get("judge") or {}
    good, bad = judge.get("good"), judge.get("bad")
    return None if good is None or bad is None else float(good - bad)


def _judges(records: list[Record]) -> list[Member]:
    """Fit judges, widest margin first. Fit is recomputed: older inventories
    stored it before the margin rule."""
    fit = [r for r in records if (_margin(r) or 0.0) >= JUDGE_FIT_MARGIN]
    margin = {r["name"]: _margin(r) or 0.0 for r in fit}
    return sorted(_members(fit), key=lambda m: (-margin[_tag(m.name)], m.name))


def _tag(name: str) -> str:
    return name.split("@")[0]


def _family(name: str) -> str:
    return name.split(":")[0]


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
    order = [h["label"] for h in inventory.get("hosts", [])]
    load: dict[str, int] = {}
    _place([m for c in cohorts for m in c.members], load, order)
    if battery.needs_judge:
        judges = _judges(inventory["models"])
        if not judges:
            raise CohortError(
                "no fit judge in this inventory; run `atomics models --probe-judge` first"
            )
        for cohort in cohorts:
            taken = {_family(m.name) for m in cohort.members}
            cohort.judge = next((j for j in judges if _family(j.name) not in taken), None)
            if cohort.judge:
                _place([cohort.judge], dict(load), order)
    return Cohorts(
        battery=battery.id,
        needs_judge=battery.needs_judge,
        taken_at=inventory.get("taken_at"),
        cohorts=cohorts,
        excluded=[Excluded(n, r) for n, r in excluded.items()],
        hosts=list(inventory.get("hosts") or []),
    )
