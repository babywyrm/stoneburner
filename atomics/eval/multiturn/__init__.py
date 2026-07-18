"""Multi-turn conversation evaluation — context retention, coherence, and instruction following."""

from __future__ import annotations

from dataclasses import dataclass, field

from atomics.models import TaskComplexity


@dataclass(frozen=True)
class ConversationTurn:
    """A single turn in a multi-turn conversation fixture."""

    user_message: str
    expected_behavior: str
    gold_criteria: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConversationFixture:
    """A multi-turn conversation evaluation fixture."""

    id: str
    complexity: TaskComplexity
    system_prompt: str
    turns: list[ConversationTurn]
    conversation_criteria: list[str] = field(default_factory=list)
    max_output_tokens: int = 512


__all__ = ["ConversationFixture", "ConversationTurn"]
