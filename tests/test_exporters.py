"""Tests for JSONL/CSV export helpers."""

import io

from atomics.exporters import write_csv, write_jsonl, write_tasks_export


def test_write_jsonl_roundtrip_keys():
    buf = io.StringIO()
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    write_jsonl(rows, buf)
    assert '"a": 1' in buf.getvalue()
    assert "\n" in buf.getvalue()


def test_write_csv_headers():
    buf = io.StringIO()
    rows = [{"z": 1, "a": 2}, {"a": 3, "z": 4}]
    write_csv(rows, buf)
    out = buf.getvalue()
    assert "a" in out and "z" in out


def test_write_tasks_export_dispatches():
    buf = io.StringIO()
    write_tasks_export([{"k": "v"}], "jsonl", buf)
    assert buf.getvalue()
