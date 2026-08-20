"""Plain-text verbose transcripts for eval fixture results."""

from __future__ import annotations

from atomics.eval.runner import FixtureResult
from atomics.models import TaskStatus


def format_eval_verbose_block(fr: FixtureResult, *, heading: bool = True) -> str:
    """Render the full prompt, reply, thinking, and judge for one fixture."""
    tr = fr.task_result
    lines: list[str] = []
    if heading:
        lines.extend([f"=== {fr.fixture.id} ({fr.fixture.complexity.value}) ===", ""])
    lines.extend(
        [
            "Prompt",
            fr.fixture.prompt,
            "",
        ]
    )
    if tr.status == TaskStatus.FAILED:
        lines.extend(["Error", tr.error_message or "(no error message)", ""])
        return "\n".join(lines).rstrip() + "\n"

    lines.extend(["Model response", tr.response or "(empty)", ""])

    if fr.thinking_text.strip():
        lines.extend(["Thinking", fr.thinking_text, ""])

    meta = []
    if fr.effort:
        meta.append(f"effort={fr.effort}")
    if fr.reasoning_mode:
        meta.append(f"reasoning_mode={fr.reasoning_mode}")
    if fr.reasoning_request:
        meta.append(f"reasoning_request={fr.reasoning_request}")
    if tr.thinking_tokens:
        meta.append(f"thinking_tokens={tr.thinking_tokens}")
    if meta:
        lines.extend(["Reasoning", "  ".join(meta), ""])

    judge = fr.judge
    lines.append("Judge")
    if judge is None:
        lines.extend(["(not scored)", ""])
        return "\n".join(lines).rstrip() + "\n"

    lines.append(
        f"score={judge.score:.3f}  ACCURACY: {judge.accuracy}/4  "
        f"COMPLETENESS: {judge.completeness}/3  FORMAT: {judge.format_score}/3"
    )
    extras = []
    if judge.parse_failed:
        extras.append("parse_failed=yes")
    if judge.criteria_coverage is not None:
        extras.append(f"criteria_coverage={judge.criteria_coverage:.3f}")
    if judge.n_judges > 1:
        extras.append(f"n_judges={judge.n_judges}")
        if judge.score_stdev is not None:
            extras.append(f"stdev={judge.score_stdev:.3f}")
    if extras:
        lines.append("  ".join(extras))
    lines.extend(["", judge.rationale or "(empty rationale)", ""])
    return "\n".join(lines).rstrip() + "\n"
