"""Split a REPL line into verb, positionals, and --flag value pairs."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field


class ParseError(ValueError):
    """The line is not verb / args / --flag value."""


_SWITCHES = frozenset({"verbose"})


@dataclass(frozen=True)
class ParsedLine:
    verb: str
    args: tuple[str, ...] = ()
    flags: dict[str, str] = field(default_factory=dict)


def parse_line(line: str) -> ParsedLine | None:
    tokens = shlex.split(line, posix=True)
    if not tokens:
        return None
    verb, rest = tokens[0], tokens[1:]
    args: list[str] = []
    flags: dict[str, str] = {}
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok.startswith("--"):
            if "=" in tok:
                name, value = tok[2:].split("=", 1)
                if not name:
                    raise ParseError(f"bad flag {tok!r}")
                flags[name] = value
                i += 1
                continue
            name = tok[2:]
            if not name:
                raise ParseError(f"bad flag {tok!r}")
            if name in _SWITCHES:
                if i + 1 < len(rest) and rest[i + 1].lower() in {"true", "false"}:
                    flags[name] = rest[i + 1].lower()
                    i += 2
                    continue
                flags[name] = "true"
                i += 1
                continue
            if i + 1 >= len(rest) or rest[i + 1].startswith("-"):
                raise ParseError(f"flag --{name} needs a value")
            flags[name] = rest[i + 1]
            i += 2
            continue
        if tok.startswith("-"):
            raise ParseError(f"unknown flag {tok!r}; use --name value")
        args.append(tok)
        i += 1
    return ParsedLine(verb=verb, args=tuple(args), flags=flags)
