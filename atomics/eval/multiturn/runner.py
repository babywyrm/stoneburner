"""Multi-turn conversation eval runner — real multi-turn with per-turn and conversation scoring."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from atomics.eval.multiturn import ConversationFixture
from atomics.eval.multiturn.fixtures import ALL_MULTITURN_FIXTURES
from atomics.eval.multiturn.judge import (
    ConversationJudgeResult,
    TurnJudgeResult,
    score_conversation,
    score_turn,
)
from atomics.models import TaskCategory, TaskResult, TaskStatus
from atomics.providers.base import BaseProvider
from atomics.validation import sanitize_error

logger = logging.getLogger("atomics.eval.multiturn.runner")


@dataclass
class TurnResult:
    turn_index: int
    user_message: str
    response: str
    latency_ms: float
    tokens: int
    cost: float
    judge: TurnJudgeResult | None


@dataclass
class ConversationResult:
    fixture: ConversationFixture
    turn_results: list[TurnResult]
    conversation_judge: ConversationJudgeResult | None
    task_result: TaskResult
    overall_score: float | None


@dataclass
class MultiturnRunSummary:
    run_id: str
    provider: str
    model: str
    judge_provider: str
    judge_model: str
    started_at: datetime
    completed_at: datetime
    conversation_results: list[ConversationResult] = field(default_factory=list)

    @property
    def avg_turn_score(self) -> float | None:
        scores: list[float] = []
        for cr in self.conversation_results:
            for tr in cr.turn_results:
                if tr.judge and not tr.judge.parse_failed:
                    scores.append(tr.judge.score)
        return round(sum(scores) / len(scores), 3) if scores else None

    @property
    def avg_conversation_score(self) -> float | None:
        scores = [
            cr.conversation_judge.score
            for cr in self.conversation_results
            if cr.conversation_judge and not cr.conversation_judge.parse_failed
        ]
        return round(sum(scores) / len(scores), 3) if scores else None

    @property
    def avg_retention(self) -> float | None:
        vals = [
            cr.conversation_judge.retention
            for cr in self.conversation_results
            if cr.conversation_judge and not cr.conversation_judge.parse_failed
        ]
        return round(sum(vals) / len(vals) / 4.0, 3) if vals else None

    @property
    def avg_consistency(self) -> float | None:
        vals = [
            cr.conversation_judge.consistency
            for cr in self.conversation_results
            if cr.conversation_judge and not cr.conversation_judge.parse_failed
        ]
        return round(sum(vals) / len(vals) / 3.0, 3) if vals else None

    @property
    def total_cost_usd(self) -> float:
        return sum(cr.task_result.estimated_cost_usd for cr in self.conversation_results)

    @property
    def avg_latency_ms(self) -> float:
        lats = [cr.task_result.latency_ms for cr in self.conversation_results if cr.task_result.latency_ms]
        return round(sum(lats) / len(lats), 1) if lats else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(cr.task_result.total_tokens for cr in self.conversation_results)

    @property
    def total_turns(self) -> int:
        return sum(len(cr.turn_results) for cr in self.conversation_results)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "model": self.model,
            "judge_provider": self.judge_provider,
            "judge_model": self.judge_model,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "avg_turn_score": self.avg_turn_score,
            "avg_conversation_score": self.avg_conversation_score,
            "avg_retention": self.avg_retention,
            "avg_consistency": self.avg_consistency,
            "total_conversations": len(self.conversation_results),
            "total_turns": self.total_turns,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_tokens": self.total_tokens,
            "conversations": [
                {
                    "id": cr.fixture.id,
                    "complexity": cr.fixture.complexity.value,
                    "num_turns": len(cr.turn_results),
                    "overall_score": cr.overall_score,
                    "conversation_judge": {
                        "retention": cr.conversation_judge.retention,
                        "consistency": cr.conversation_judge.consistency,
                        "instruction": cr.conversation_judge.instruction,
                        "score": cr.conversation_judge.score,
                        "rationale": cr.conversation_judge.rationale,
                    } if cr.conversation_judge and not cr.conversation_judge.parse_failed else None,
                    "turns": [
                        {
                            "turn": t.turn_index,
                            "score": t.judge.score if t.judge else None,
                            "latency_ms": t.latency_ms,
                            "tokens": t.tokens,
                        }
                        for t in cr.turn_results
                    ],
                }
                for cr in self.conversation_results
            ],
        }


def _build_transcript(
    system_prompt: str,
    turns: list[tuple[str, str]],
) -> str:
    """Build a transcript string from system prompt and completed turns."""
    parts = [f"[System]: {system_prompt}"]
    for user_msg, assistant_msg in turns:
        parts.append(f"[User]: {user_msg}")
        parts.append(f"[Assistant]: {assistant_msg}")
    return "\n\n".join(parts)


async def run_multiturn(
    provider: BaseProvider,
    *,
    judge_provider: BaseProvider | None = None,
    model: str | None = None,
    judge_model: str | None = None,
    run_id: str | None = None,
    on_conversation_done: Callable[[ConversationResult], None] | None = None,
    thinking: bool | None = None,
    thinking_budget: int | None = None,
    fixtures: list[ConversationFixture] | None = None,
) -> MultiturnRunSummary:
    """Run the multi-turn conversation evaluation."""
    effective_judge = judge_provider or provider
    effective_run_id = run_id or uuid.uuid4().hex[:12]
    selected = fixtures or ALL_MULTITURN_FIXTURES
    started = datetime.now(UTC)

    conversation_results: list[ConversationResult] = []

    for fixture in selected:
        completed_turns: list[tuple[str, str]] = []
        turn_results: list[TurnResult] = []
        total_tokens = 0
        total_cost = 0.0
        total_latency = 0.0
        conversation_failed = False

        for i, turn in enumerate(fixture.turns):
            transcript = _build_transcript(fixture.system_prompt, completed_turns)
            prompt = turn.user_message
            if completed_turns:
                prompt = f"{transcript}\n\n[User]: {turn.user_message}"

            gen_kwargs: dict = {
                "system": fixture.system_prompt if not completed_turns else "",
                "model": model,
                "max_tokens": fixture.max_output_tokens,
            }
            if thinking is not None:
                gen_kwargs["thinking"] = thinking
            if thinking_budget is not None:
                gen_kwargs["thinking_budget"] = thinking_budget

            try:
                resp = await provider.generate(prompt, **gen_kwargs)
                response_text = resp.text
                turn_latency = resp.latency_ms
                turn_tokens = resp.total_tokens
                turn_cost = resp.estimated_cost_usd
            except Exception as exc:
                response_text = ""
                turn_latency = 0.0
                turn_tokens = 0
                turn_cost = 0.0
                conversation_failed = True
                logger.warning("Turn %d of %s failed: %s", i, fixture.id, sanitize_error(exc))

            completed_turns.append((turn.user_message, response_text))
            total_tokens += turn_tokens
            total_cost += turn_cost
            total_latency += turn_latency

            if response_text and not conversation_failed:
                full_transcript = _build_transcript(fixture.system_prompt, completed_turns)
                turn_judge = await score_turn(
                    transcript=full_transcript,
                    user_message=turn.user_message,
                    response=response_text,
                    expected_behavior=turn.expected_behavior,
                    judge=effective_judge,
                    judge_model=judge_model,
                )
            else:
                turn_judge = None

            turn_results.append(TurnResult(
                turn_index=i,
                user_message=turn.user_message,
                response=response_text,
                latency_ms=turn_latency,
                tokens=turn_tokens,
                cost=turn_cost,
                judge=turn_judge,
            ))

            if conversation_failed:
                break

        full_transcript = _build_transcript(fixture.system_prompt, completed_turns)
        if not conversation_failed:
            conv_judge = await score_conversation(
                transcript=full_transcript,
                criteria=fixture.conversation_criteria,
                judge=effective_judge,
                judge_model=judge_model,
            )
        else:
            conv_judge = None

        turn_scores = [
            t.judge.score for t in turn_results
            if t.judge and not t.judge.parse_failed
        ]
        conv_score = conv_judge.score if conv_judge and not conv_judge.parse_failed else None
        if turn_scores and conv_score is not None:
            overall = round((sum(turn_scores) / len(turn_scores) + conv_score) / 2, 3)
        elif turn_scores:
            overall = round(sum(turn_scores) / len(turn_scores), 3)
        else:
            overall = None

        tr = TaskResult(
            task_id=uuid.uuid4().hex[:12],
            run_id=effective_run_id,
            category=TaskCategory.GENERAL_QA,
            task_name=fixture.id,
            provider=provider.name,
            model=model or "",
            status=TaskStatus.FAILED if conversation_failed else TaskStatus.SUCCESS,
            prompt=full_transcript,
            response="\n---\n".join(t.response for t in turn_results),
            total_tokens=total_tokens,
            latency_ms=total_latency,
            estimated_cost_usd=total_cost,
            accuracy_score=overall,
            started_at=started,
            completed_at=datetime.now(UTC),
        )

        cr = ConversationResult(
            fixture=fixture,
            turn_results=turn_results,
            conversation_judge=conv_judge,
            task_result=tr,
            overall_score=overall,
        )
        conversation_results.append(cr)
        if on_conversation_done:
            on_conversation_done(cr)

    return MultiturnRunSummary(
        run_id=effective_run_id,
        provider=provider.name,
        model=model or getattr(provider, "default_model", None) or "",
        judge_provider=effective_judge.name,
        judge_model=judge_model or "",
        started_at=started,
        completed_at=datetime.now(UTC),
        conversation_results=conversation_results,
    )
