"""Security-focused CLI commands.

Each command lives in its own `cmd_<name>` submodule; this package re-exports
them so `from atomics.commands.security import adversarial` and
`from atomics.commands import security` both keep working. The submodules carry
the `cmd_` prefix so re-exporting the command function does not shadow the
module it came from — tests still patch `security.cmd_<name>.<helper>`.
"""

# `_parse_model_spec` and `_make_provider` are re-exported because atomics.cli
# still aliases them (and a few tests import atomics.cli._parse_model_spec); the
# noqa marks them as deliberate re-exports rather than unused imports.
from atomics.commands.common import _make_provider  # noqa: F401
from atomics.commands.security.cmd_adversarial import (  # noqa: F401
    _parse_model_spec,
    adversarial,
)
from atomics.commands.security.cmd_codereview import codereview
from atomics.commands.security.cmd_multiturn import multiturn
from atomics.commands.security.cmd_redblue import redblue
from atomics.commands.security.cmd_refusal import refusal

__all__ = ["adversarial", "codereview", "multiturn", "redblue", "refusal"]
