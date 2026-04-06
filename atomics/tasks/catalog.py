"""Task catalog with combinatoric prompt generation.

Each task type has multiple phrasing templates. The selection pipeline:
  1. Filter tasks by tier complexity
  2. Weighted random task selection
  3. Pick a topic from the deep pool (recency-checked)
  4. Pick a random phrasing template for that task
  5. Optionally cross-pollinate with a secondary topic
  6. Layer random modifiers (audience, constraint, format, perspective)
"""

from __future__ import annotations

import random

from atomics.models import (
    TIER_COMPLEXITY_MAP,
    BurnTier,
    TaskCategory,
    TaskComplexity,
    TaskDefinition,
)
from atomics.tasks.randomizer import (
    build_combinatoric_prompt,
    build_modified_prompt,
    get_recency_tracker,
)
from atomics.tasks.topics import CROSS_POLLINATION_TOPICS, TOPIC_POOLS

# Phrasing variants per task — {topic} is the only required placeholder
PHRASING_VARIANTS: dict[str, list[str]] = {
    # ── LIGHT ─────────────────────────────────────────────
    "quick_question": [
        "Answer this concisely: {topic}",
        "Give a brief, precise answer: {topic}",
        "In a few sentences, explain: {topic}",
        "Quick take: {topic}",
        "ELI-senior-engineer: {topic}",
    ],
    "definition_lookup": [
        "Define {topic} in 2-3 sentences for a technical audience.",
        "What is {topic}? Be precise and technical.",
        "Give a rigorous but concise definition of {topic}.",
        "How would you define {topic} in a security context?",
    ],
    "summarize_tech_article": [
        "Summarize the key points of {topic} in 3-5 bullet points.",
        "Give a senior engineer's briefing on {topic}.",
        "What are the essential takeaways from {topic}?",
        "Distill {topic} into its core concepts and trade-offs.",
        "Write an executive summary of {topic} for a technical audience.",
    ],
    # ── MODERATE ──────────────────────────────────────────
    "explain_concept": [
        "Explain {topic} clearly and concisely for a technical audience.",
        "Break down {topic} — what is it, why does it matter, and where does it fail?",
        "Walk through {topic} as if teaching it to a strong junior engineer.",
        "Provide a technical deep-read on {topic} with concrete examples.",
    ],
    "compare_technologies": [
        "Compare and contrast {topic} with its main alternatives.",
        "What are the trade-offs between {topic} and competing approaches?",
        "Rank the alternatives to {topic} by security posture and performance.",
        "When would you choose {topic} over the alternatives, and vice versa?",
    ],
    "cve_analysis": [
        "Analyze the security implications of {topic}.",
        "Threat-model {topic}: impact, affected systems, mitigations.",
        "Write a security advisory for {topic} including detection strategies.",
        "How would you detect and respond to exploitation of {topic}?",
    ],
    "extract_patterns": [
        "Identify recurring patterns and anti-patterns in {topic}.",
        "What are the best practices and common pitfalls for {topic}?",
        "Extract the design patterns that emerge from {topic}.",
        "Catalog the failure modes and success patterns in {topic}.",
    ],
    "code_review_snippet": [
        "Review the following concept and suggest security improvements: {topic}",
        "What would a thorough code review focus on for {topic}?",
        "Identify potential vulnerabilities in typical implementations of {topic}.",
        "How would you harden an implementation of {topic}?",
    ],
    # ── HEAVY ─────────────────────────────────────────────
    "deep_dive_topic": [
        (
            "Provide a detailed technical analysis of {topic} including architecture, "
            "trade-offs, and a comparison matrix of at least 3 alternatives."
        ),
        (
            "Write a comprehensive technical report on {topic} covering design "
            "decisions, failure modes, and operational considerations."
        ),
        (
            "Deep-dive into {topic}: cover the theory, practical implementation, "
            "performance characteristics, and security implications."
        ),
    ],
    "threat_landscape": [
        (
            "Write a comprehensive threat intelligence brief on {topic} with MITRE "
            "ATT&CK mapping, detection strategies, and prioritized mitigations."
        ),
        (
            "Analyze the full threat landscape around {topic}: actors, TTPs, affected "
            "systems, and defense recommendations."
        ),
        (
            "Produce a threat assessment for {topic} including attack trees, "
            "probability estimates, and recommended security controls."
        ),
    ],
    "architecture_review": [
        (
            "Design a production architecture for {topic} with component diagrams, "
            "data flow, failure modes, and scaling strategy."
        ),
        (
            "Architect a secure, scalable system for {topic} — cover trust boundaries, "
            "data classification, and operational runbooks."
        ),
        (
            "Create an architectural decision record for {topic} evaluating 3+ "
            "approaches with security and performance trade-offs."
        ),
    ],
    "security_audit_patterns": [
        (
            "Conduct a security audit analysis of {topic}: attack surfaces, "
            "vulnerability categories, remediation priorities, and testing strategy."
        ),
        (
            "Enumerate the attack surface of {topic} and propose a layered defense "
            "strategy with detection and response procedures."
        ),
        (
            "Perform a systematic security review of {topic} covering injection, "
            "auth, crypto, config, and supply chain risks."
        ),
    ],
    "multi_step_reasoning": [
        (
            "Walk through {topic} step-by-step, showing your reasoning at each stage "
            "with edge cases and alternatives."
        ),
        (
            "Solve {topic} methodically: state assumptions, derive the approach, "
            "analyze complexity, and verify correctness."
        ),
        (
            "Apply first-principles reasoning to {topic} — decompose the problem, "
            "evaluate approaches, and construct a rigorous solution."
        ),
    ],
}


TASK_CATALOG: list[TaskDefinition] = [
    # ── LIGHT ─────────────────────────────────────────────
    TaskDefinition(
        category=TaskCategory.GENERAL_QA,
        name="quick_question",
        prompt_template="{prompt}",
        complexity=TaskComplexity.LIGHT,
        weight=3.0,
        max_output_tokens=256,
    ),
    TaskDefinition(
        category=TaskCategory.GENERAL_QA,
        name="definition_lookup",
        prompt_template="{prompt}",
        complexity=TaskComplexity.LIGHT,
        weight=2.5,
        max_output_tokens=128,
    ),
    TaskDefinition(
        category=TaskCategory.WEB_SUMMARY,
        name="summarize_tech_article",
        prompt_template="{prompt}",
        complexity=TaskComplexity.LIGHT,
        weight=2.0,
        max_output_tokens=512,
    ),
    # ── MODERATE ──────────────────────────────────────────
    TaskDefinition(
        category=TaskCategory.WEB_SUMMARY,
        name="explain_concept",
        prompt_template="{prompt}",
        complexity=TaskComplexity.MODERATE,
        weight=1.5,
        max_output_tokens=768,
    ),
    TaskDefinition(
        category=TaskCategory.RESEARCH,
        name="compare_technologies",
        prompt_template="{prompt}",
        complexity=TaskComplexity.MODERATE,
        weight=1.5,
        max_output_tokens=1024,
    ),
    TaskDefinition(
        category=TaskCategory.SECURITY_NEWS,
        name="cve_analysis",
        prompt_template="{prompt}",
        complexity=TaskComplexity.MODERATE,
        weight=2.0,
        max_output_tokens=1024,
    ),
    TaskDefinition(
        category=TaskCategory.PATTERN_EXTRACTION,
        name="extract_patterns",
        prompt_template="{prompt}",
        complexity=TaskComplexity.MODERATE,
        weight=1.5,
        max_output_tokens=768,
    ),
    TaskDefinition(
        category=TaskCategory.GENERAL_QA,
        name="code_review_snippet",
        prompt_template="{prompt}",
        complexity=TaskComplexity.MODERATE,
        weight=1.5,
        max_output_tokens=512,
    ),
    # ── HEAVY ─────────────────────────────────────────────
    TaskDefinition(
        category=TaskCategory.RESEARCH,
        name="deep_dive_topic",
        prompt_template="{prompt}",
        complexity=TaskComplexity.HEAVY,
        weight=1.0,
        max_output_tokens=2048,
    ),
    TaskDefinition(
        category=TaskCategory.SECURITY_NEWS,
        name="threat_landscape",
        prompt_template="{prompt}",
        complexity=TaskComplexity.HEAVY,
        weight=1.0,
        max_output_tokens=2048,
    ),
    TaskDefinition(
        category=TaskCategory.RESEARCH,
        name="architecture_review",
        prompt_template="{prompt}",
        complexity=TaskComplexity.HEAVY,
        weight=0.8,
        max_output_tokens=3000,
    ),
    TaskDefinition(
        category=TaskCategory.PATTERN_EXTRACTION,
        name="security_audit_patterns",
        prompt_template="{prompt}",
        complexity=TaskComplexity.HEAVY,
        weight=0.8,
        max_output_tokens=3000,
    ),
    TaskDefinition(
        category=TaskCategory.GENERAL_QA,
        name="multi_step_reasoning",
        prompt_template="{prompt}",
        complexity=TaskComplexity.HEAVY,
        weight=1.0,
        max_output_tokens=2048,
    ),
]


def _pick_topic(category: TaskCategory, tracker) -> str:
    """Pick a topic that hasn't been used recently."""
    pool = TOPIC_POOLS.get(category, ["general software engineering"])
    random.shuffle(pool)
    for topic in pool:
        if not tracker.is_recent(topic):
            tracker.record(topic)
            return topic
    # Fallback: all topics are recent, just pick one
    topic = random.choice(pool)
    tracker.record(topic)
    return topic


def get_weighted_task(tier: BurnTier = BurnTier.BASELINE) -> tuple[TaskDefinition, str]:
    """Select a random task and generate a unique prompt via combinatorics."""
    tracker = get_recency_tracker()

    allowed = TIER_COMPLEXITY_MAP[tier]
    eligible = [t for t in TASK_CATALOG if t.complexity in allowed]
    if not eligible:
        eligible = [t for t in TASK_CATALOG if t.complexity == TaskComplexity.LIGHT]

    weights = [t.weight for t in eligible]
    task = random.choices(eligible, weights=weights, k=1)[0]

    topic = _pick_topic(task.category, tracker)

    # Pick a phrasing variant
    variants = PHRASING_VARIANTS.get(task.name, ["{topic}"])
    template = random.choice(variants)
    base_prompt = template.format(topic=topic)

    # 25% chance of cross-pollination for moderate+ tasks
    if task.complexity != TaskComplexity.LIGHT and random.random() < 0.25:
        secondary = random.choice(CROSS_POLLINATION_TOPICS)
        base_prompt = build_combinatoric_prompt(topic, secondary)

    # Layer modifiers — heavier tasks get more
    prompt = build_modified_prompt(
        base_prompt,
        add_audience=True,
        add_constraint=task.complexity != TaskComplexity.LIGHT,
        add_format=task.complexity == TaskComplexity.HEAVY,
        add_perspective=task.complexity != TaskComplexity.LIGHT,
    )

    return task, prompt
