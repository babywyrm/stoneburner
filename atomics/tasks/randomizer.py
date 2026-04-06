"""Combinatoric prompt randomizer.

Generates unique prompts by combining:
  - Multiple phrasing templates per task type
  - Deep topic pools with cross-category mixing
  - Random modifiers (audience, constraints, format)
  - Recency tracking to avoid repeats within a window
"""

from __future__ import annotations

import hashlib
import random
from collections import deque
from dataclasses import dataclass, field


@dataclass
class PromptModifier:
    audience: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    formats: list[str] = field(default_factory=list)
    perspectives: list[str] = field(default_factory=list)


AUDIENCES = [
    "for a senior security engineer",
    "for a principal architect reviewing a design doc",
    "for a SRE on-call at 3am",
    "for a pentester writing a report",
    "for a developer new to distributed systems",
    "for a CISO preparing a board presentation",
    "for a grad student studying cryptography",
    "for someone migrating from monolith to microservices",
    "for a red team operator",
    "for a platform engineering team",
    "for a compliance auditor",
    "for a kernel developer",
]

CONSTRAINTS = [
    "Keep it under 300 words.",
    "Include at least one concrete code example.",
    "Reference specific CVEs or RFCs where applicable.",
    "Include a threat model perspective.",
    "Assume zero-trust network architecture.",
    "Consider both cloud-native and on-prem deployments.",
    "Account for FIPS 140-2 compliance requirements.",
    "Prioritize by blast radius.",
    "Include time complexity analysis where relevant.",
    "Assume adversarial conditions.",
    "Consider supply chain implications.",
    "Frame in terms of MITRE ATT&CK techniques.",
]

FORMATS = [
    "Structure your response as a decision matrix.",
    "Use a numbered priority list.",
    "Format as an architectural decision record (ADR).",
    "Present as a risk assessment with likelihood and impact.",
    "Structure as attack tree branches.",
    "Format as a runbook with conditional steps.",
    "Present as a comparison table with trade-offs.",
    "Write as a brief threat intelligence report.",
    "Structure as a proof sketch.",
    "Format as a security advisory.",
]

PERSPECTIVES = [
    "from a defense-in-depth standpoint",
    "through the lens of the principle of least privilege",
    "considering Byzantine fault tolerance",
    "from an attacker's perspective",
    "with emphasis on formal verification",
    "considering information-theoretic security",
    "from a chaos engineering mindset",
    "through the lens of queueing theory",
    "considering post-quantum readiness",
    "from an observability-first perspective",
    "with a focus on deterministic reproducibility",
    "through the lens of game theory",
]

# Cross-pollination prefixes that combine two domains
COMBINATORS = [
    "In the context of {secondary}, analyze {primary}.",
    "Compare how {primary} and {secondary} intersect in practice.",
    "Explain {primary} using analogies from {secondary}.",
    "What lessons from {secondary} apply to {primary}?",
    "How would a deep understanding of {secondary} change your approach to {primary}?",
    "Where does {primary} break down when combined with {secondary}?",
    "Design a system that must handle both {primary} and {secondary} simultaneously.",
    "What are the emergent properties when {primary} meets {secondary}?",
]


class RecencyTracker:
    """Tracks recently used topics/prompts to avoid repetition."""

    def __init__(self, window: int = 50) -> None:
        self._seen: deque[str] = deque(maxlen=window)

    def is_recent(self, key: str) -> bool:
        h = hashlib.md5(key.encode()).hexdigest()[:10]
        return h in self._seen

    def record(self, key: str) -> None:
        h = hashlib.md5(key.encode()).hexdigest()[:10]
        self._seen.append(h)

    @property
    def seen_count(self) -> int:
        return len(self._seen)


# Singleton tracker shared across the engine lifetime
_recency = RecencyTracker(window=100)


def get_recency_tracker() -> RecencyTracker:
    return _recency


def build_modified_prompt(
    base_prompt: str,
    *,
    add_audience: bool = True,
    add_constraint: bool = True,
    add_format: bool = False,
    add_perspective: bool = True,
) -> str:
    """Layer random modifiers onto a base prompt."""
    parts = [base_prompt]

    if add_audience and random.random() < 0.6:
        parts.append(random.choice(AUDIENCES))

    if add_perspective and random.random() < 0.5:
        parts.append(random.choice(PERSPECTIVES))

    if add_constraint and random.random() < 0.4:
        parts.append(random.choice(CONSTRAINTS))

    if add_format and random.random() < 0.3:
        parts.append(random.choice(FORMATS))

    return " ".join(parts)


def build_combinatoric_prompt(primary: str, secondary: str) -> str:
    """Combine two topics using a random combinator template."""
    template = random.choice(COMBINATORS)
    return template.format(primary=primary, secondary=secondary)
