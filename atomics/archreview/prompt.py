"""Analysis prompt + findings parser for archreview."""

from __future__ import annotations

import re

from atomics.archreview.models import Finding
from atomics.archreview.taxonomy import ALL_CATEGORIES, normalize_category

_SYSTEM = (
    "You are a senior security architect performing a code review. "
    "Be precise and specific; cite file locations. Do not invent issues."
)

_TASK_TEMPLATE = """\
Review the repository below for security-architecture weaknesses.

First give a SHORT architecture & trust-boundary summary (3-5 sentences).

Then list findings. Emit EACH finding on its own line in EXACTLY this format:
CATEGORY: <one of: {categories}> | LOCATION: <file or area> | SEVERITY: <low|medium|high|critical> | WHY: <1-2 sentences>

Use only categories from the list. One line per finding. Do not number them.

REPOSITORY:
{pack}
"""

# Strict: a CATEGORY/LOCATION/SEVERITY/WHY line in canonical order.
_LINE_RE = re.compile(
    r"CATEGORY\W{0,4}(?P<cat>[^|\n]+?)\s*[|]\s*"
    r"LOCATION\W{0,4}(?P<loc>[^|\n]+?)\s*[|]\s*"
    r"SEVERITY\W{0,4}(?P<sev>[^|\n]+?)\s*[|]\s*"
    r"WHY\W{0,4}(?P<why>[^\n]+)",
    re.IGNORECASE,
)

# Lenient: same four fields in any order, markdown-tolerant, separators free.
_FIELD_CAT = re.compile(r"category\W{0,6}([^|\n*]+)", re.IGNORECASE)
_FIELD_LOC = re.compile(r"location\W{0,6}([^|\n*]+)", re.IGNORECASE)
_FIELD_SEV = re.compile(r"severity\W{0,6}([^|\n*]+)", re.IGNORECASE)
_FIELD_WHY = re.compile(r"why\W{0,6}([^|\n*]+)", re.IGNORECASE)
_PIPE_RE = re.compile(
    r"^\s*(?P<cat>[a-z][a-z0-9_ -]{1,80})\s*[|]\s*"
    r"(?P<loc>[^|\n]{1,200})\s*[|]\s*"
    r"(?P<sev>low|medium|high|critical)\s*[|]\s*"
    r"(?P<why>.+?)\s*$",
    re.IGNORECASE,
)


def build_analysis_prompt(pack_text: str) -> tuple[str, str]:
    cats = ", ".join(c.value for c in ALL_CATEGORIES)
    task = _TASK_TEMPLATE.format(categories=cats, pack=pack_text)
    return _SYSTEM, task


def _clean(s: str) -> str:
    return s.strip().strip("*_-").strip()


def _mk_finding(cat_raw: str, loc: str, sev: str, why: str) -> Finding:
    norm = normalize_category(_clean(cat_raw))
    return Finding(
        category=norm.value if norm else "unknown",
        location=_clean(loc),
        severity=_clean(sev).lower(),
        rationale=_clean(why),
    )


def parse_findings(raw: str) -> list[Finding]:
    """Parse a model's analysis into findings. Strict pass, then lenient."""
    findings: list[Finding] = []
    for m in _LINE_RE.finditer(raw):
        findings.append(_mk_finding(m.group("cat"), m.group("loc"),
                                    m.group("sev"), m.group("why")))
    if findings:
        return findings

    # Lenient: scan line-by-line for any line carrying all four fields.
    for line in raw.splitlines():
        if "category" not in line.lower():
            continue
        cat, loc, sev, why = (_FIELD_CAT.search(line), _FIELD_LOC.search(line),
                              _FIELD_SEV.search(line), _FIELD_WHY.search(line))
        if cat and loc and sev and why:
            findings.append(_mk_finding(cat.group(1), loc.group(1),
                                        sev.group(1), why.group(1)))
    if findings:
        return findings

    # Common near-miss: models preserve the pipe-delimited field order but drop
    # labels, e.g. "injection | routes/search.ts | high | raw SQL".
    for line in raw.splitlines():
        m = _PIPE_RE.match(line)
        if not m:
            continue
        findings.append(_mk_finding(m.group("cat"), m.group("loc"),
                                    m.group("sev"), m.group("why")))
    return findings
