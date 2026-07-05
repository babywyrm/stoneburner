"""Tests for labcompare orchestration helpers."""
from __future__ import annotations

import pytest

from atomics.labcompare import HostSpec, parse_host_specs


def test_parse_single_host():
    specs = parse_host_specs(["laptop=http://192.168.1.205:11434"])
    assert specs == [HostSpec(name="laptop", url="http://192.168.1.205:11434")]


def test_parse_multiple_hosts():
    specs = parse_host_specs([
        "laptop=http://192.168.1.205:11434",
        "brainbox=http://192.168.1.239:11434",
    ])
    assert len(specs) == 2
    assert specs[1].name == "brainbox"


def test_parse_host_missing_equals_raises():
    with pytest.raises(ValueError, match="expected NAME=URL"):
        parse_host_specs(["http://192.168.1.205:11434"])


def test_parse_host_empty_name_raises():
    with pytest.raises(ValueError, match="empty host name"):
        parse_host_specs(["=http://x:11434"])
