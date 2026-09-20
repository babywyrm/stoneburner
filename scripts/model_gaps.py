"""Compare installed Ollama tags with recent public releases.

Usage: uv run python scripts/model_gaps.py [--days 45] [--json]

Reads /api/tags on the laptop, beefy, and brainbox. Polls the Ollama
library page and recent Hugging Face text models from the labs this
tool already scores. Prints sizes the catalog advertises that at least
one host does not have. Does not pull. Does not load a model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Any

import httpx

from atomics.benchmark.model_classes import (
    supports_ollama_think_field,
    supports_ollama_think_levels,
)

DEFAULT_HOSTS: dict[str, str] = {
    "laptop": "http://127.0.0.1:11434",
    "beefy": "http://beefy:11434",
    "brainbox": "http://brainbox:11434",
}

HF_ORGS: tuple[str, ...] = (
    "Qwen",
    "google",
    "meta-llama",
    "mistralai",
    "microsoft",
    "nvidia",
    "deepseek-ai",
    "ibm-granite",
    "LiquidAI",
)
HF_PIPELINES = frozenset({"text-generation", "image-text-to-text", "any-to-any"})
_DATE_RE = re.compile(r'title="([A-Z][a-z]{2} \d{1,2}, \d{4})')
_LIB_RE = re.compile(r'href="/library/([a-z0-9][a-z0-9._-]*)"')
_BADGE_RE = re.compile(r"bg-\[#ddf4ff\][^>]*>\s*([^<]+?)\s*<")
_SIZE_RE = re.compile(r"-((?:e)?\d+(?:\.\d+)?b)$")
_QUANT_RE = re.compile(r"-(nvfp4|mxfp4|fp8|fp16|bf16|gguf|awq|gptq)$")


@dataclass(frozen=True)
class Remote:
    source: str
    name: str
    sizes: tuple[str, ...]
    updated: date
    thinking: bool


@dataclass(frozen=True)
class Gap:
    source: str
    name: str
    size: str
    updated: str
    missing_on: tuple[str, ...]
    registry: str


def known_family(name: str) -> bool:
    """True when the think registry already has a rule for this family."""
    return supports_ollama_think_levels(name) or not supports_ollama_think_field(name)


def family_of(tag: str) -> str:
    return tag.split(":", 1)[0].lower()


def size_of(tag: str) -> str:
    if ":" not in tag:
        return ""
    return tag.split(":", 1)[1].split("-", 1)[0].lower()


def hf_name_and_size(repo: str) -> tuple[str, str]:
    """Map a Hub repo id onto an Ollama family and size, when the name allows it."""
    leaf = repo.split("/")[-1].lower()
    while True:
        stripped = _QUANT_RE.sub("", leaf)
        if stripped == leaf:
            break
        leaf = stripped
    leaf = re.sub(r"-(it|instruct|base|preview)$", "", leaf)
    match = _SIZE_RE.search(leaf)
    if match is None:
        return leaf, ""
    size = match.group(1)
    family = leaf[: match.start()]
    parts = family.split("-")
    if len(parts) == 2 and parts[1][:1].isdigit():
        family = parts[0] + parts[1]
    return family, size


def host_has(tags: set[str], name: str, size: str) -> bool:
    want_family = name.lower()
    want_size = size.lower()
    for tag in tags:
        if family_of(tag) != want_family:
            continue
        if not want_size or size_of(tag) == want_size:
            return True
    return False


def parse_ollama_library(html: str, *, today: date, days: int) -> list[Remote]:
    """Recent library cards. A card with no absolute date is skipped."""
    cutoff = today - timedelta(days=days)
    found: list[Remote] = []
    for card in re.split(r"<li\b", html):
        link = _LIB_RE.search(card)
        dated = _DATE_RE.search(card)
        if link is None or dated is None:
            continue
        updated = datetime.strptime(dated.group(1), "%b %d, %Y").date()
        if updated < cutoff:
            continue
        sizes = tuple(dict.fromkeys(part.strip().lower() for part in _BADGE_RE.findall(card)))
        thinking = "thinking" in card
        found.append(
            Remote(
                source="ollama",
                name=link.group(1),
                sizes=sizes,
                updated=updated,
                thinking=thinking,
            )
        )
    return found


def parse_hf_models(payload: list[dict[str, Any]], *, today: date, days: int) -> list[Remote]:
    cutoff = today - timedelta(days=days)
    found: list[Remote] = []
    for item in payload:
        pipeline = item.get("pipeline_tag")
        if pipeline not in HF_PIPELINES:
            continue
        raw = str(item.get("lastModified") or "")
        if len(raw) < 10:
            continue
        updated = date.fromisoformat(raw[:10])
        if updated < cutoff:
            continue
        name, size = hf_name_and_size(str(item["id"]))
        found.append(
            Remote(
                source="huggingface",
                name=name,
                sizes=(size,) if size else (),
                updated=updated,
                thinking=known_family(name),
            )
        )
    return found


def gaps(installed: dict[str, set[str]], remotes: list[Remote]) -> list[Gap]:
    """Sizes at least one live host does not have.

    A host that returned no catalog is omitted. It is down, not empty.
    """
    rows: list[Gap] = []
    for remote in remotes:
        sizes = remote.sizes or ("",)
        for size in sizes:
            missing = tuple(
                host
                for host, tags in installed.items()
                if not host_has(tags, remote.name, size)
            )
            if not missing:
                continue
            rows.append(
                Gap(
                    source=remote.source,
                    name=remote.name,
                    size=size,
                    updated=remote.updated.isoformat(),
                    missing_on=missing,
                    registry="known" if known_family(remote.name) else "new-family",
                )
            )
    rows.sort(key=lambda row: (row.updated, row.name, row.size), reverse=True)
    seen: set[tuple[str, str, str]] = set()
    unique: list[Gap] = []
    for row in rows:
        key = (row.source, row.name, row.size)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def fetch_tags(client: httpx.Client, base: str) -> set[str]:
    response = client.get(f"{base.rstrip('/')}/api/tags")
    response.raise_for_status()
    payload = response.json()
    return {str(item["name"]) for item in payload.get("models", [])}


def fetch_library(client: httpx.Client, *, today: date, days: int) -> list[Remote]:
    response = client.get("https://ollama.com/library")
    response.raise_for_status()
    return parse_ollama_library(response.text, today=today, days=days)


def fetch_hf(client: httpx.Client, *, today: date, days: int) -> list[Remote]:
    found: list[Remote] = []
    for org in HF_ORGS:
        response = client.get(
            "https://huggingface.co/api/models",
            params={"author": org, "sort": "lastModified", "direction": "-1", "limit": 15},
        )
        response.raise_for_status()
        found.extend(parse_hf_models(response.json(), today=today, days=days))
    return found


def _print(installed: dict[str, set[str]], down: dict[str, str], rows: list[Gap]) -> None:
    for host, tags in installed.items():
        print(f"{host}: {len(tags)} tags")
    for host, err in down.items():
        print(f"{host}: down ({err})")
    nowhere = [row for row in rows if len(row.missing_on) == len(installed)]
    partial = [row for row in rows if len(row.missing_on) != len(installed)]
    print(f"\nnot on any host ({len(nowhere)})")
    for row in nowhere:
        _line(row)
    print(f"\nmissing on some hosts ({len(partial)})")
    for row in partial:
        _line(row)


def _line(row: Gap) -> None:
    size = f":{row.size}" if row.size else ""
    where = ", ".join(row.missing_on)
    print(f"  {row.source} {row.name}{size}  {row.updated}  {row.registry}  missing {where}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=45)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    today = date.today()
    installed: dict[str, set[str]] = {}
    down: dict[str, str] = {}
    timeout = httpx.Timeout(20.0, connect=8.0)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for host, base in DEFAULT_HOSTS.items():
            try:
                installed[host] = fetch_tags(client, base)
            except (httpx.HTTPError, OSError) as exc:
                down[host] = exc.__class__.__name__
        try:
            remotes = fetch_library(client, today=today, days=args.days)
        except (httpx.HTTPError, OSError) as exc:
            print(f"ollama library: {exc.__class__.__name__}", file=sys.stderr)
            remotes = []
        try:
            remotes.extend(fetch_hf(client, today=today, days=args.days))
        except (httpx.HTTPError, OSError) as exc:
            print(f"huggingface: {exc.__class__.__name__}", file=sys.stderr)
    rows = gaps(installed, remotes)
    if args.json:
        json.dump([asdict(row) for row in rows], sys.stdout, indent=2)
        print()
        return 0
    _print(installed, down, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
