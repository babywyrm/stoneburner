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

Then list findings. Output EACH finding on its own line using EXACTLY this \
format — no markdown, no numbering, no table, no extra punctuation:
CATEGORY: <value> | LOCATION: <file or area> | SEVERITY: <low|medium|high|critical> | WHY: <1-2 sentences>

Valid CATEGORY values: {categories}

Example (do not copy — write real findings):
CATEGORY: injection | LOCATION: routes/login.ts | SEVERITY: critical | WHY: unsanitized user input passed directly to SQL query.

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
_HYBRID_PIPE_RE = re.compile(
    r"^\s*(?P<cat>[a-z][a-z0-9_ -]{1,80})\s*[|]\s*"
    r"(?:location|file|route|area)\s*[:=-]?\s*(?P<loc>[^|\n]{1,200})\s*[|]\s*"
    r"severity\s*[:=-]?\s*(?P<sev>low|medium|high|critical)\s*[|]\s*"
    r"(?:why|rationale)\s*[:=-]?\s*(?P<why>.+?)\s*$",
    re.IGNORECASE,
)
# Markdown table row: | cat | location | severity | why |
# Skips separator rows like |---|---|---|---| and header rows whose first cell
# isn't a recognizable category.
_MD_TABLE_ROW_RE = re.compile(
    r"^\s*[|]\s*(?P<cat>[a-z][a-z0-9_ -]{1,80})\s*[|]\s*"
    r"(?P<loc>[^|\n]{1,200})\s*[|]\s*"
    r"(?P<sev>low|medium|high|critical)\s*[|]\s*"
    r"(?P<why>[^|\n]+?)\s*[|]?\s*$",
    re.IGNORECASE,
)
# Numbered/bulleted bold list: 1. **Category** — location — severity — why
_BOLD_LIST_RE = re.compile(
    r"^\s*(?:\d+\.|[-*])\s+\*\*(?P<cat>[^*]+)\*\*\s*[—\-]\s*"
    r"(?P<loc>[^—\-\n]+?)\s*[—\-]\s*"
    r"(?P<sev>low|medium|high|critical)\s*[—\-]\s*"
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


def _deduplicate(findings: list[Finding]) -> list[Finding]:
    """Remove exact (category, location) duplicates, preserving first occurrence."""
    seen: set[tuple[str, str]] = set()
    out: list[Finding] = []
    for f in findings:
        key = (f.category, f.location.strip().lower()[:80])
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def parse_findings(raw: str) -> list[Finding]:
    """Parse a model's analysis into findings. Strict pass, then lenient."""
    findings: list[Finding] = []
    for m in _LINE_RE.finditer(raw):
        # Skip markdown table header rows — _LINE_RE matches "CATEGORY | LOCATION
        # | SEVERITY | WHY" as a finding with blank/whitespace fields.
        cat_raw = m.group("cat").strip()
        loc_raw = m.group("loc").strip()
        if not cat_raw or not loc_raw:
            continue
        findings.append(_mk_finding(cat_raw, loc_raw,
                                    m.group("sev"), m.group("why")))
    if findings:
        return _deduplicate(findings)

    # Lenient: scan line-by-line for any line carrying all four fields.
    for line in raw.splitlines():
        if "category" not in line.lower():
            continue
        cat, loc, sev, why = (_FIELD_CAT.search(line), _FIELD_LOC.search(line),
                              _FIELD_SEV.search(line), _FIELD_WHY.search(line))
        if cat and loc and sev and why:
            # Skip header rows where 'category' is the label, not a category value.
            if normalize_category(_clean(cat.group(1))) is None:
                continue
            findings.append(_mk_finding(cat.group(1), loc.group(1),
                                        sev.group(1), why.group(1)))
    if findings:
        return _deduplicate(findings)

    # Common near-misses: models preserve pipe-delimited field order but drop
    # some or all labels, e.g. "injection | routes/search.ts | high | raw SQL"
    # or "INJECTION | ROUTE: routes/login.ts | SEVERITY: high | WHY: raw SQL".
    for line in raw.splitlines():
        if re.match(r"^\s*[|][\s\-:|]+[|]", line):
            continue  # markdown separator row
        m = _HYBRID_PIPE_RE.match(line) or _PIPE_RE.match(line)
        if not m:
            continue
        # Skip header rows: first cell is a label word (Category, Finding, etc.)
        # not a recognizable category value.
        if normalize_category(_clean(m.group("cat"))) is None:
            continue
        findings.append(_mk_finding(m.group("cat"), m.group("loc"),
                                    m.group("sev"), m.group("why")))
    if findings:
        return _deduplicate(findings)

    # Markdown table rows: | cat | location | severity | why |
    # Skip separator rows (only dashes/spaces) and header rows (first cell
    # not a recognizable category).
    for line in raw.splitlines():
        if re.match(r"^\s*[|][\s\-:|]+[|]", line):
            continue  # separator/header row
        m = _MD_TABLE_ROW_RE.match(line)
        if not m:
            continue
        cat_norm = normalize_category(_clean(m.group("cat")))
        if cat_norm is None:
            continue  # skip header rows whose first cell isn't a category
        findings.append(Finding(
            category=cat_norm.value,
            location=_clean(m.group("loc")),
            severity=_clean(m.group("sev")).lower(),
            rationale=_clean(m.group("why")),
        ))
    if findings:
        return findings

    # Numbered/bulleted bold list: 1. **Category** — location — severity — why
    for line in raw.splitlines():
        m = _BOLD_LIST_RE.match(line)
        if not m:
            continue
        findings.append(_mk_finding(m.group("cat"), m.group("loc"),
                                    m.group("sev"), m.group("why")))
    return _deduplicate(findings)
