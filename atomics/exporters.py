"""JSONL / CSV export for task metrics."""

from __future__ import annotations

import csv
import json
from typing import Any, TextIO


def _json_default(obj: Any) -> str:
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def write_jsonl(rows: list[dict], out: TextIO) -> None:
    for row in rows:
        out.write(json.dumps(row, default=_json_default) + "\n")


def write_csv(rows: list[dict], out: TextIO) -> None:
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    w = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)


def write_tasks_export(rows: list[dict], fmt: str, out: TextIO) -> None:
    if fmt == "jsonl":
        write_jsonl(rows, out)
    else:
        write_csv(rows, out)
