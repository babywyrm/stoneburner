"""Every atomics command in QUICKSTART.md must be one the CLI accepts.

The test parses. It does not run the command, so it never calls a model.
A missing file in an example path is allowed. An unknown command or flag
is not.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import click
import pytest

from atomics.cli import cli

_QUICKSTART = Path(__file__).resolve().parents[1] / "QUICKSTART.md"
_PATH_MISS = ("does not exist", "is a directory", "is not writable", "not found")


def _fences(text: str) -> list[str]:
    blocks: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("```bash"):
            i += 1
            body: list[str] = []
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            blocks.append("\n".join(body))
        i += 1
    return blocks


def _commands(block: str) -> list[str]:
    joined: list[str] = []
    buf = ""
    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            buf += line[:-1].strip() + " "
            continue
        buf += line
        joined.append(buf.strip())
        buf = ""
    if buf.strip():
        joined.append(buf.strip())
    return joined


def _drop_comment(line: str) -> str:
    out: list[str] = []
    quote = ""
    i = 0
    while i < len(line):
        char = line[i]
        if quote:
            if char == quote:
                quote = ""
            out.append(char)
        elif char in "\"'":
            quote = char
            out.append(char)
        elif char == "#" and (i == 0 or line[i - 1] == " "):
            break
        else:
            out.append(char)
        i += 1
    return "".join(out).strip()


def _atomics_argv(line: str) -> list[str] | None:
    parts = shlex.split(_drop_comment(line), posix=True)
    while parts and "=" in parts[0] and not parts[0].startswith("-"):
        parts.pop(0)
    if parts[:2] == ["uv", "run"]:
        parts = parts[2:]
    if not parts or parts[0] != "atomics":
        return None
    return parts[1:]


def _documented() -> list[list[str]]:
    argv_list: list[list[str]] = []
    for block in _fences(_QUICKSTART.read_text(encoding="utf-8")):
        for line in _commands(block):
            argv = _atomics_argv(line)
            if argv is not None:
                argv_list.append(argv)
    return argv_list


def test_quickstart_documents_commands() -> None:
    assert len(_documented()) >= 40


@pytest.mark.parametrize("argv", _documented(), ids=lambda argv: " ".join(argv)[:80])
def test_quickstart_command_parses(argv: list[str]) -> None:
    try:
        cli.make_context("atomics", argv)
    except click.UsageError as exc:
        message = str(exc).lower()
        if any(hint in message for hint in _PATH_MISS):
            return
        pytest.fail(f"{' '.join(argv)}\n{exc}")
