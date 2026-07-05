"""Tests for labcompare orchestration helpers."""
from __future__ import annotations

import pytest

from atomics.labcompare import HostSpec, parse_host_specs


def test_parse_single_host():
    specs = parse_host_specs(["laptop=http://192.0.2.14:11434"])
    assert specs == [HostSpec(name="laptop", url="http://192.0.2.14:11434")]


def test_parse_multiple_hosts():
    specs = parse_host_specs([
        "laptop=http://192.0.2.14:11434",
        "gpu-host-b=http://192.0.2.16:11434",
    ])
    assert len(specs) == 2
    assert specs[1].name == "gpu-host-b"


def test_parse_host_missing_equals_raises():
    with pytest.raises(ValueError, match="expected NAME=URL"):
        parse_host_specs(["http://192.0.2.14:11434"])


def test_parse_host_empty_name_raises():
    with pytest.raises(ValueError, match="empty host name"):
        parse_host_specs(["=http://x:11434"])
