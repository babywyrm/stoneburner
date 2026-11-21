"""Security-focused CLI commands.

Each command lives in its own `cmd_<name>` submodule; this package re-exports
them so `from atomics.commands.security import adversarial` and
`from atomics.commands import security` both keep working. The submodules carry
the `cmd_` prefix so re-exporting the command function does not shadow the
module it came from — tests still patch `security.cmd_<name>.<helper>`.
"""

from atomics.commands.security.cmd_adversarial import adversarial
from atomics.commands.security.cmd_codereview import codereview
from atomics.commands.security.cmd_multiturn import multiturn
from atomics.commands.security.cmd_redblue import redblue
from atomics.commands.security.cmd_refusal import refusal

__all__ = ["adversarial", "codereview", "multiturn", "redblue", "refusal"]
