"""Named security batteries — jobs, not new suites.

Each battery is a documented subset of runners that already exist.
`atomics battery show` prints the commands. `run` is a later increment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CostBand = Literal["cheap", "medium", "heavy"]


@dataclass(frozen=True)
class BatteryStep:
    suite: str
    purpose: str
    fixtures: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    category: str | None = None
    mode: str | None = None
    qa_file: str | None = None
    repo: str | None = None
    extra: tuple[str, ...] = ()


@dataclass(frozen=True)
class Battery:
    id: str
    title: str
    purpose: str
    when: str
    not_a_pass: str
    label_hint: str
    cost_band: CostBand
    needs_judge: bool
    steps: tuple[BatteryStep, ...]
    optional_steps: tuple[BatteryStep, ...] = ()


BATTERIES: tuple[Battery, ...] = (
    Battery(
        id="desk-pass",
        title="Desk pass",
        purpose="Health, app-gate patterns, and a two-fixture tool-channel taste.",
        when="New tag, laptop, or first hour on a box. Minutes, not overnight.",
        not_a_pass=(
            "Not walkthrough-compatible. Not a resilience number. "
            "Tool probe skip is not refusal."
        ),
        label_hint="FUNCTION_COMPATIBLE if provider-test and qa hold; tool-capable must be honest",
        cost_band="cheap",
        needs_judge=False,
        steps=(
            BatteryStep(
                suite="provider-test",
                purpose="Runtime health only",
            ),
            BatteryStep(
                suite="qa",
                purpose="App-gate pattern contract",
                qa_file="qa/examples/app-gate-guardrails.yaml",
            ),
            BatteryStep(
                suite="toolcall",
                purpose="Shadow vs rm — one dangerous, one no-call is a finding not a pass",
                fixtures=("tc-01", "tc-02"),
                extra=("--channel", "tools", "--no-save"),
            ),
        ),
    ),
    Battery(
        id="blue-capability",
        title="Blue capability",
        purpose="Can this model do SOC work: logs, IR, hardening, STRIDE, Sigma, code review.",
        when="Choosing a defender / analyst copilot. Needs a judge.",
        not_a_pass="Not resilience. A 90% blue score can still be DANGEROUS on tc-01.",
        label_hint="capability axis — pair with agent-gate before putting it in front of tools",
        cost_band="medium",
        needs_judge=True,
        steps=(
            BatteryStep(
                suite="redblue",
                purpose="Five blue fixtures",
                mode="blue",
                extra=("--no-save",),
            ),
            BatteryStep(
                suite="codereview",
                purpose="Planted CWE vs clean false positives",
                extra=("--no-save",),
            ),
            BatteryStep(
                suite="refusal",
                purpose="Over-refusal on legitimate defensive questions",
                fixtures=("rc-b01", "rc-b02", "rc-b03", "rc-b04", "rc-b05", "rc-b06"),
                extra=("--no-save",),
            ),
        ),
    ),
    Battery(
        id="red-capability",
        title="Red capability",
        purpose="Conceptual offensive reasoning: OSINT, vuln class, privesc, AD, web.",
        when="Choosing an authorized-pentest copilot. Needs a judge.",
        not_a_pass=(
            "Not a green light to run exploits. Do not add ransomware / phishing "
            "operational fixtures here — those belong in refusal as should-refuse."
        ),
        label_hint="capability axis — conceptual only",
        cost_band="medium",
        needs_judge=True,
        steps=(
            BatteryStep(
                suite="redblue",
                purpose="Five red fixtures",
                mode="red",
                extra=("--no-save",),
            ),
            BatteryStep(
                suite="refusal",
                purpose="Must still answer authorized dual-use questions",
                fixtures=("rc-b05",),
                extra=("--no-save",),
            ),
        ),
    ),
    Battery(
        id="agent-gate",
        title="Agent gate",
        purpose="Prose resistance on MCP/tool-safety/tool-desc plus a short tool-channel pack.",
        when="The model will be handed tools or MCP. Score the two channels separately.",
        not_a_pass=(
            "Prose refusal is not a tool-channel refusal. "
            "North-style probe skip is not resistance."
        ),
        label_hint="resilience + tool channel; UNSAFE_GATE_BEHAVIOR if DANGEROUS on direct tools",
        cost_band="medium",
        needs_judge=True,
        steps=(
            BatteryStep(
                suite="adversarial",
                purpose="mcp + tool_safety + tool_desc_injection (21 fixtures, not 72)",
                categories=("mcp", "tool_safety", "tool_desc_injection"),
                extra=("--no-save",),
            ),
            BatteryStep(
                suite="toolcall",
                purpose="direct category, tools channel, no judge required for this step",
                category="direct",
                extra=("--channel", "tools", "--no-save"),
            ),
        ),
    ),
    Battery(
        id="threat-model",
        title="Threat model / architecture",
        purpose="STRIDE for an agentic system plus agentic architecture reasoning.",
        when="The question is 'does this model reason about agent security architecture'.",
        not_a_pass=(
            "One STRIDE prompt is not a threat-model practice. "
            "archreview is optional and needs a repo pack."
        ),
        label_hint="capability axis — architectural reasoning, not gate behavior",
        cost_band="medium",
        needs_judge=True,
        steps=(
            BatteryStep(
                suite="redblue",
                purpose="Existing STRIDE agent threat model",
                fixtures=("rb-b04",),
                extra=("--no-save",),
            ),
            BatteryStep(
                suite="adversarial",
                purpose="agentic reasoning (ar-01..ar-11)",
                categories=("agentic",),
                extra=("--no-save",),
            ),
        ),
        optional_steps=(
            BatteryStep(
                suite="archreview",
                purpose="Repo evidence pack vs OWASP key — skip without JUICE_SHOP_PATH",
                repo="juice-shop",
                extra=("--tier", "floor"),
            ),
        ),
    ),
)

_BY_ID: dict[str, Battery] = {b.id: b for b in BATTERIES}


def get_battery(name: str) -> Battery:
    try:
        return _BY_ID[name]
    except KeyError as exc:
        known = ", ".join(b.id for b in BATTERIES)
        raise KeyError(f"unknown battery {name!r}. Known: {known}") from exc


_JUDGED = frozenset({"adversarial", "redblue", "refusal", "codereview", "archreview"})
_BUDGET_SUITES = frozenset(
    {"adversarial", "redblue", "refusal", "codereview", "toolcall", "archreview"}
)
_NO_THINKING_SUITES = frozenset(
    {
        "provider-test",
        "qa",
        "toolcall",
        "adversarial",
        "redblue",
        "refusal",
        "codereview",
    }
)


def visible_steps(
    battery: Battery,
    *,
    provider: str,
    profile: str | None = None,
) -> tuple[BatteryStep, ...]:
    """Raw qa is Ollama HTTP. Other providers skip it unless an app-gate profile is set."""
    out: list[BatteryStep] = []
    for step in battery.steps:
        if step.suite == "qa" and provider != "ollama" and not profile:
            continue
        out.append(step)
    return tuple(out)


def _route_flags(
    *,
    provider: str,
    model: str | None,
    ollama_host: str | None,
    vllm_host: str | None,
) -> list[str]:
    args = ["--provider", provider]
    if model:
        args.extend(["-m", model])
    if ollama_host:
        args.extend(["--ollama-host", ollama_host])
    if vllm_host:
        args.extend(["--vllm-host", vllm_host])
    return args


def step_args(
    step: BatteryStep,
    *,
    model: str | None,
    provider: str = "ollama",
    thinking_flag: str = "--no-thinking",
    ollama_host: str | None = None,
    vllm_host: str | None = None,
    judge_provider: str | None = None,
    judge_model: str | None = None,
    judge_host: str | None = None,
    budget: str | None = None,
    profile: str | None = None,
) -> list[str]:
    """Argv for one step, starting with the subcommand. No `atomics` prefix.

    Omit ``model`` so a matching ``inference.env`` can fill it. Do not invent a
    provider — the caller passes ``-p``.
    """
    if step.suite == "provider-test":
        args = ["provider-test", "-p", provider]
        if thinking_flag:
            args.append(thinking_flag)
        if model:
            args.extend(["-m", model])
        if ollama_host:
            args.extend(["--ollama-host", ollama_host])
        if vllm_host:
            args.extend(["--vllm-host", vllm_host])
        return args
    if step.suite == "qa":
        if profile:
            args = ["qa", "--profile", profile]
            if thinking_flag:
                args.append(thinking_flag)
            if model:
                args.extend(["-m", model])
            return args
        if not step.qa_file:
            raise ValueError("qa step needs qa_file")
        args = ["qa", "--file", step.qa_file]
        if thinking_flag:
            args.append(thinking_flag)
        if model:
            args.extend(["-m", model])
        if ollama_host:
            args.extend(["--ollama-host", ollama_host])
        return args
    if step.suite == "archreview":
        if not step.repo:
            raise ValueError("archreview step needs repo")
        args = ["archreview", "--repo", step.repo, "--provider", provider]
        if model:
            args.extend(["--models", model])
        args.extend(step.extra)
        if judge_provider:
            args.extend(["--judge-provider", judge_provider])
        if judge_model:
            args.extend(["--judge-model", judge_model])
        if budget:
            args.extend(["--budget", budget])
        return args

    args = [step.suite]
    args.extend(
        _route_flags(
            provider=provider,
            model=model,
            ollama_host=ollama_host,
            vllm_host=vllm_host,
        )
    )
    if thinking_flag and step.suite in _NO_THINKING_SUITES:
        args.append(thinking_flag)
    if step.suite == "redblue" and step.mode:
        args.extend(["--mode", step.mode])
    if step.suite == "adversarial" and step.categories:
        args.extend(["--category", ",".join(step.categories)])
    if step.category:
        args.extend(["--category", step.category])
    if step.fixtures:
        args.extend(["--fixtures", ",".join(step.fixtures)])
    if step.suite in _JUDGED:
        if judge_provider:
            args.extend(["--judge-provider", judge_provider])
        if judge_model:
            args.extend(["--judge-model", judge_model])
        if judge_host:
            args.extend(["--judge-host", judge_host])
    if budget and step.suite in _BUDGET_SUITES:
        args.extend(["--budget", budget])
    args.extend(step.extra)
    return args
