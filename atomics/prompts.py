"""Catalog of the built-in prompts each suite and judge sends.

The strings live beside the code that sends them; this module only collects
them so `atomics prompts` can show them and `--show-prompt` can name the
system prompt on each call. Profile and multiturn fixture prompts are
per-fixture data, not defaults, so they are not listed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptEntry:
    name: str
    role: str  # "model" (the model under test) or "judge"
    system: str
    template: str | None
    used_by: str


def catalog() -> list[PromptEntry]:
    from atomics.archreview import prompt as archreview_prompt
    from atomics.archreview import scorer as archreview_scorer
    from atomics.core import runner as core_runner
    from atomics.eval import judge
    from atomics.eval import runner as eval_runner
    from atomics.eval.adversarial import runner as adversarial_runner
    from atomics.eval.adversarial import scorer as adversarial_scorer
    from atomics.eval.codegen import runner as codegen_runner
    from atomics.eval.codereview import runner as codereview_runner
    from atomics.eval.codereview import scorer as codereview_scorer
    from atomics.eval.multiturn import judge as multiturn_judge
    from atomics.eval.rag import judge as rag_judge
    from atomics.eval.redblue import runner as redblue_runner
    from atomics.eval.refusal import runner as refusal_runner
    from atomics.eval.refusal import scorer as refusal_scorer
    from atomics.eval.toolcall import runner as toolcall_runner
    from atomics.probe import runner as probe_runner

    return [
        PromptEntry("eval", "model", eval_runner._SYSTEM_PROMPT, None, "eval, sweep"),
        PromptEntry(
            "eval.judge",
            "judge",
            judge._JUDGE_SYSTEM,
            judge._RUBRIC_TEMPLATE,
            "eval, sweep, redblue, probe",
        ),
        PromptEntry(
            "eval.judge-reformat",
            "judge",
            judge._REFORMAT_SYSTEM,
            judge._REFORMAT_TEMPLATE,
            "retry when a judge reply does not parse",
        ),
        PromptEntry("redblue", "model", redblue_runner._SYSTEM_PROMPT, None, "redblue"),
        PromptEntry("refusal", "model", refusal_runner._SYSTEM_PROMPT, None, "refusal"),
        PromptEntry(
            "refusal.judge", "judge", refusal_scorer._SYSTEM, refusal_scorer._TEMPLATE, "refusal"
        ),
        PromptEntry("adversarial", "model", adversarial_runner._SYSTEM_PROMPT, None, "adversarial"),
        PromptEntry(
            "adversarial.judge",
            "judge",
            adversarial_scorer._SYSTEM,
            adversarial_scorer._RESIST_TEMPLATE,
            "adversarial, toolcall",
        ),
        PromptEntry("toolcall", "model", toolcall_runner._SYSTEM_PROMPT, None, "toolcall"),
        PromptEntry(
            "codereview",
            "model",
            codereview_runner._REVIEW_SYSTEM,
            codereview_runner._REVIEW_TEMPLATE,
            "codereview",
        ),
        PromptEntry(
            "codereview.judge",
            "judge",
            codereview_scorer._SYSTEM,
            codereview_scorer._VULNERABLE_TEMPLATE,
            "codereview, vulnerable samples",
        ),
        PromptEntry(
            "codereview.judge-clean",
            "judge",
            codereview_scorer._SYSTEM,
            codereview_scorer._CLEAN_TEMPLATE,
            "codereview, clean samples",
        ),
        PromptEntry(
            "multiturn.judge-turn",
            "judge",
            multiturn_judge._TURN_JUDGE_SYSTEM,
            multiturn_judge._TURN_RUBRIC_TEMPLATE,
            "multiturn",
        ),
        PromptEntry(
            "multiturn.judge-conversation",
            "judge",
            multiturn_judge._CONV_JUDGE_SYSTEM,
            multiturn_judge._CONV_RUBRIC_TEMPLATE,
            "multiturn",
        ),
        PromptEntry(
            "rag.judge",
            "judge",
            rag_judge._RAG_JUDGE_SYSTEM,
            rag_judge._RAG_RUBRIC_TEMPLATE,
            "rag",
        ),
        PromptEntry(
            "codegen",
            "model",
            codegen_runner._CODEGEN_SYSTEM,
            codegen_runner._CODEGEN_PROMPT,
            "codegen",
        ),
        PromptEntry(
            "archreview",
            "model",
            archreview_prompt._SYSTEM,
            archreview_prompt._TASK_TEMPLATE,
            "archreview",
        ),
        PromptEntry(
            "archreview.judge",
            "judge",
            archreview_scorer._REASONING_SYSTEM,
            archreview_scorer._REASONING_TEMPLATE,
            "archreview",
        ),
        PromptEntry("probe", "model", probe_runner._SYSTEM_PROMPT, None, "probe"),
        PromptEntry(
            "benchmark", "model", core_runner._SYSTEM_PROMPT, None, "run, distributed workers"
        ),
    ]


def names_for_system(system: str) -> list[str]:
    """Catalog names whose system prompt is exactly `system`."""
    if not system:
        return []
    return [e.name for e in catalog() if e.system == system]
