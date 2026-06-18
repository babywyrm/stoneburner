from atomics.archreview.taxonomy import Category, normalize_category, ALL_CATEGORIES


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
