from atomics.archreview.taxonomy import ALL_CATEGORIES, Category, normalize_category


def test_all_categories_are_enum_values():
    assert Category.INJECTION in ALL_CATEGORIES
    assert all(isinstance(c, Category) for c in ALL_CATEGORIES)


def test_normalize_exact_enum_value():
    assert normalize_category("broken_access_control") == Category.BROKEN_ACCESS_CONTROL


def test_normalize_synonyms_and_casing():
    assert normalize_category("Broken Access Control") == Category.BROKEN_ACCESS_CONTROL
    assert normalize_category("SQLi") == Category.INJECTION
    assert normalize_category("cross-site scripting") == Category.XSS
    assert normalize_category("sensitive data exposure") == Category.SENSITIVE_DATA_EXPOSURE


def test_normalize_unmappable_returns_none():
    assert normalize_category("definitely not a category") is None
    assert normalize_category("") is None


def test_normalize_non_web_synonyms():
    """Common findings in non-web repos (Go APIs, IaC, Rust services) should map."""
    assert normalize_category("path traversal") == Category.BROKEN_ACCESS_CONTROL
    assert normalize_category("directory traversal") == Category.BROKEN_ACCESS_CONTROL
    assert normalize_category("hardcoded credentials") == Category.SENSITIVE_DATA_EXPOSURE
    assert normalize_category("hardcoded secret") == Category.SENSITIVE_DATA_EXPOSURE
    assert normalize_category("race condition") == Category.IMPROPER_INPUT_VALIDATION
    assert normalize_category("privilege escalation") == Category.BROKEN_ACCESS_CONTROL
    assert normalize_category("open redirect") == Category.BROKEN_ACCESS_CONTROL
    assert normalize_category("unvalidated redirect") == Category.BROKEN_ACCESS_CONTROL
    assert normalize_category("weak cipher") == Category.CRYPTOGRAPHIC_ISSUES
    assert normalize_category("tls misconfiguration") == Category.SECURITY_MISCONFIGURATION
    assert normalize_category("dependency confusion") == Category.VULNERABLE_COMPONENTS
    assert normalize_category("supply chain") == Category.VULNERABLE_COMPONENTS
    assert normalize_category("log injection") == Category.INJECTION
    assert normalize_category("header injection") == Category.INJECTION
