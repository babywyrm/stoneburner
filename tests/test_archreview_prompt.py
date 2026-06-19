from atomics.archreview.prompt import build_analysis_prompt, parse_findings


def test_prompt_includes_pack_and_taxonomy():
    sys_p, task_p = build_analysis_prompt("PACK-CONTENT-HERE")
    assert "security architect" in sys_p.lower()
    assert "PACK-CONTENT-HERE" in task_p
    assert "broken_access_control" in task_p  # taxonomy enumerated


def test_parse_strict_block():
    raw = (
        "Summary: trust boundaries are weak.\n"
        "CATEGORY: injection | LOCATION: routes/search.ts | SEVERITY: high | "
        "WHY: unsanitized query passed to db.\n"
        "CATEGORY: xss | LOCATION: views/p.html | SEVERITY: medium | "
        "WHY: reflected user input.\n"
    )
    findings = parse_findings(raw)
    cats = {f.category for f in findings}
    assert cats == {"injection", "xss"}
    assert findings[0].location == "routes/search.ts"


def test_parse_tolerates_markdown_and_reordering():
    raw = (
        "- **LOCATION**: a.ts — **CATEGORY**: SQL Injection — "
        "**SEVERITY**: high — **WHY**: concatenated SQL\n"
    )
    findings = parse_findings(raw)
    assert len(findings) == 1
    assert findings[0].category == "injection"


def test_parse_unmappable_category_marked_unknown():
    raw = "CATEGORY: frobnication | LOCATION: x | SEVERITY: low | WHY: n/a\n"
    findings = parse_findings(raw)
    assert findings[0].category == "unknown"


def test_parse_pipe_delimited_without_labels():
    raw = (
        "## Findings\n"
        "injection | routes/search.ts | high | unsanitized query passed to db.\n"
        "broken access control | routes/users.ts | medium | missing authorization.\n"
    )
    findings = parse_findings(raw)
    assert [f.category for f in findings] == ["injection", "broken_access_control"]
    assert findings[0].location == "routes/search.ts"


def test_parse_pipe_delimited_with_labeled_location_severity_why():
    raw = (
        "SECURITY_MISCONFIGURATION | FILE: .well-known/security.txt | "
        "SEVERITY: low | WHY: Contact email is publicly exposed.\n"
        "INJECTION | ROUTE: routes/login.ts (line 173) | "
        "SEVERITY: critical | WHY: String interpolation in SQL query.\n"
    )
    findings = parse_findings(raw)
    assert [f.category for f in findings] == ["security_misconfiguration", "injection"]
    assert findings[0].location == ".well-known/security.txt"
    assert findings[1].location == "routes/login.ts (line 173)"
    assert findings[1].severity == "critical"


def test_parse_markdown_table_rows():
    """Models that default to markdown tables should be parsed."""
    raw = (
        "## Findings\n"
        "| injection | routes/search.ts | high | unsanitized query passed to db |\n"
        "| broken_access_control | routes/users.ts | medium | missing authorization check |\n"
    )
    findings = parse_findings(raw)
    assert [f.category for f in findings] == ["injection", "broken_access_control"]
    assert findings[0].severity == "high"


def test_parse_markdown_table_with_header():
    """Full markdown table including a header/separator row."""
    raw = (
        "| Category | Location | Severity | Why |\n"
        "|---|---|---|---|\n"
        "| xss | views/user.html | medium | reflected input |\n"
        "| injection | routes/login.ts | critical | raw SQL |\n"
    )
    findings = parse_findings(raw)
    cats = {f.category for f in findings}
    assert "xss" in cats
    assert "injection" in cats


def test_parse_numbered_bold_list():
    """1. **Category** — location — severity — why"""
    raw = (
        "1. **Injection** — routes/login.ts — high — unsanitized SQL\n"
        "2. **XSS** — views/profile.html — medium — reflected input\n"
    )
    findings = parse_findings(raw)
    assert len(findings) == 2
    assert findings[0].category == "injection"
    assert findings[1].category == "xss"


def test_parse_deduplicates_same_category_location():
    """Duplicate (category, location) pairs from looping models are collapsed."""
    raw = "\n".join([
        "CATEGORY: improper_input_validation | LOCATION: routes/basketItems.ts | SEVERITY: medium | WHY: repeated finding."
    ] * 7)
    findings = parse_findings(raw)
    assert len(findings) == 1
    assert findings[0].category == "improper_input_validation"


def test_parse_keeps_same_category_different_locations():
    """Same category at different locations are distinct findings."""
    raw = (
        "CATEGORY: injection | LOCATION: routes/login.ts | SEVERITY: high | WHY: a.\n"
        "CATEGORY: injection | LOCATION: routes/search.ts | SEVERITY: high | WHY: b.\n"
    )
    findings = parse_findings(raw)
    assert len(findings) == 2


def test_parse_empty_returns_empty_list():
    assert parse_findings("no findings here, all good") == []
