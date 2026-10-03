"""Rich table for an inventory."""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from atomics.inventory import Capability, Inventory, ModelRecord

_CLASS_STYLE = {"light": "green", "mid": "yellow", "heavy": "red"}
_SHOWN = ("tools", "thinking", "vision")


def _cap_cell(cap: Capability) -> str:
    if cap.value is None:
        return "[dim]?[/dim]"
    word = "[green]yes[/green]" if cap.value else "[dim]no[/dim]"
    return word if cap.source == "declared" else f"{word} [dim]{cap.source}[/dim]"


def _context_cell(m: ModelRecord) -> str:
    text = str(m.declared_context) if m.declared_context else "?"
    if m.requested_context:
        text += f" → {m.requested_context}"
    if m.loaded_context:
        text += f" (loaded {m.loaded_context})"
    return text


def _answer_cell(m: ModelRecord) -> str:
    if m.probe is None:
        return ""
    off = m.probe.off
    if off is None:
        return "[red]error[/red]"
    word = "[green]yes[/green]" if off.answered else f"[red]{off.outcome}[/red]"
    return f"{word} {off.tokens_per_second:.0f} t/s" if off.tokens_per_second else word


def _switch_cell(m: ModelRecord) -> str:
    if m.probe is None or m.probe.verdict is None:
        return ""
    return f"{m.probe.verdict} → {m.probe.recommended}"


def _notes(m: ModelRecord) -> list[str]:
    evaluable = [] if m.evaluable else ["not evaluable: no completion capability"]
    return [*evaluable, *m.disagreements, *m.flags]


def render(inv: Inventory, console: Console) -> None:
    for host in inv.hosts:
        line = f"[bold]{escape(host.label)}[/bold] {host.provider} {host.version or ''}".rstrip()
        if host.loaded:
            line += " — loaded: " + ", ".join(f"{escape(n)} ({c})" for n, c in host.loaded)
        console.print(f"{line} [red]{escape(host.error)}[/red]" if host.error else line)

    probed = any(m.probe for m in inv.models)
    table = Table(box=box.SIMPLE_HEAD)
    longest = max((len(m.name) for m in inv.models), default=5)
    table.add_column("Model", style="cyan bold", no_wrap=True, min_width=longest)
    table.add_column("Size", justify="right", no_wrap=True)
    table.add_column("Params", justify="right", no_wrap=True)
    table.add_column("Quant", no_wrap=True)
    table.add_column("Context", justify="right", no_wrap=True)
    table.add_column("Class", no_wrap=True)
    for cap in _SHOWN:
        table.add_column(cap.capitalize(), justify="center", no_wrap=True)
    if probed:
        table.add_column("Answer", no_wrap=True)
        table.add_column("Thinking switch", no_wrap=True)
    for m in sorted(inv.models, key=lambda m: (m.host, m.size_bytes or 0)):
        style = _CLASS_STYLE.get(m.model_class, "dim")
        row = [
            escape(m.name),
            f"{m.size_bytes / 1e9:.1f} GB" if m.size_bytes else "",
            m.parameter_size or "",
            m.quantization or "",
            _context_cell(m),
            f"[{style}]{m.model_class}[/{style}]",
            *(_cap_cell(m.capability(c)) for c in _SHOWN),
        ]
        if probed:
            row += [_answer_cell(m), _switch_cell(m)]
        table.add_row(*row)
    console.print(table)

    for m in inv.models:
        for note in _notes(m):
            console.print(f"[yellow]{escape(m.name)}[/yellow] {note}")
        for error in m.errors:
            console.print(f"[red]{escape(m.name)}[/red] {escape(error)}")
    unknown = sum(1 for m in inv.models if m.model_class == "unknown")
    if unknown:
        console.print(
            f"\n[yellow]{unknown} unregistered model(s) — "
            f"add to model_classes.py for accurate comparison[/yellow]"
        )
