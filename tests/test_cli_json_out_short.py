"""Every --json-out takes -o, so one habit works on every command."""

import click

from atomics.cli import cli


def _commands(group: click.Group, prefix: str = ""):
    for name, cmd in group.commands.items():
        if isinstance(cmd, click.Group):
            yield from _commands(cmd, f"{prefix}{name} ")
        else:
            yield f"{prefix}{name}", cmd


def test_every_json_out_has_the_short_flag():
    missing = [
        name
        for name, cmd in _commands(cli)
        for p in cmd.params
        if "--json-out" in getattr(p, "opts", []) and "-o" not in p.opts
    ]
    assert missing == []
