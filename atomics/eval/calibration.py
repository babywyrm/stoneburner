"""Judge calibration: verify the quality judge ranks graded answers correctly.

A judge is only trustworthy if a clearly-better answer reliably scores higher
than a clearly-worse one. This module ships a small set of fixtures, each with
several answers ordered worst → best, and a harness that scores them and checks
that the resulting scores are monotonically non-decreasing and well separated.

Used as a regression guard (see tests/test_calibration.py): a judge-prompt or
parsing change that breaks ranking will fail the calibration check.

`run_judge_calibration` is the absolute check: a human-graded set scored on the
judge's own rubric, reported as mean error and bias against the human.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import itertools
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from inspect import isawaitable
from pathlib import Path
from typing import Literal

import yaml

from atomics.eval.adversarial.scorer import score_resistance
from atomics.eval.codereview.fixtures import SecureCodeFixture
from atomics.eval.codereview.scorer import _score_for_verdict, judge_review
from atomics.eval.judge import JudgeResult, char_budget_for_tokens, score_response
from atomics.eval.outcomes import JudgeOutcomeStatus, RunIntegrity, RunStatus
from atomics.eval.refusal.scorer import classification_to_score, classify_response
from atomics.eval.suite_integrity import fixture_outcome, integrity_of
from atomics.providers.base import BaseProvider
from atomics.storage import MetricsRepository
from atomics.storage.records import EvaluationResultRecord
from atomics.validation import sanitize_error

logger = logging.getLogger(__name__)

# Signature of a scoring function compatible with score_response.
ScoreFn = Callable[..., Awaitable[JudgeResult]]


@dataclass(frozen=True)
class CalibrationCase:
    """A prompt with answers ordered from worst (index 0) to best (index -1)."""

    id: str
    prompt: str
    graded_answers: tuple[str, ...]
    gold_criteria: tuple[str, ...] = ()


# Each case's answers are deliberately separated by quality: an off-topic/wrong
# answer, a thin-but-correct answer, and a thorough, technically precise answer.
CALIBRATION_CASES: tuple[CalibrationCase, ...] = (
    CalibrationCase(
        id="cal-hash",
        prompt="What is a cryptographic hash function and name one security property?",
        graded_answers=(
            "A hash function is a kind of breakfast you make with potatoes.",
            "A hash function maps data to a fixed-size value.",
            (
                "A cryptographic hash function deterministically maps arbitrary-length "
                "input to a fixed-size digest. Key security properties include "
                "preimage resistance, second-preimage resistance, and collision "
                "resistance, so it is infeasible to invert or find colliding inputs."
            ),
        ),
        gold_criteria=("fixed-size digest", "collision resistance", "preimage resistance"),
    ),
    CalibrationCase(
        id="cal-tcp",
        prompt="Explain what TCP provides to applications.",
        graded_answers=(
            "TCP is a popular brand of coffee enjoyed worldwide.",
            "TCP is a networking protocol that sends data between computers.",
            (
                "TCP is a connection-oriented transport protocol that provides "
                "reliable, ordered, error-checked delivery of a byte stream. It uses "
                "a three-way handshake, sequence numbers and acknowledgements, "
                "retransmission of lost segments, and flow and congestion control."
            ),
        ),
        gold_criteria=("connection-oriented", "reliable", "congestion control"),
    ),
)


@dataclass
class CaseCalibration:
    case_id: str
    scores: list[float]
    monotonic: bool  # scores never decrease from worst → best
    separated: bool  # best - worst >= min_separation


@dataclass
class CalibrationReport:
    results: list[CaseCalibration] = field(default_factory=list)
    min_separation: float = 0.2

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(r.monotonic and r.separated for r in self.results)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        ok = sum(1 for r in self.results if r.monotonic and r.separated)
        return round(ok / len(self.results), 3)


def _is_monotonic(scores: Sequence[float]) -> bool:
    return all(b >= a for a, b in zip(scores, scores[1:]))


async def calibrate_judge(
    judge_provider: BaseProvider,
    *,
    judge_model: str | None = None,
    cases: Sequence[CalibrationCase] = CALIBRATION_CASES,
    min_separation: float = 0.2,
    score_fn: ScoreFn = score_response,
) -> CalibrationReport:
    """Score each case's graded answers and check ranking quality.

    Returns a CalibrationReport whose ``passed`` is True only when every case is
    ranked monotonically (worst → best never decreases) and the best answer beats
    the worst by at least ``min_separation`` on the 0.0-1.0 scale.
    """
    report = CalibrationReport(min_separation=min_separation)
    for case in cases:
        scores: list[float] = []
        for answer in case.graded_answers:
            result = await score_fn(
                case.prompt,
                answer,
                judge_provider=judge_provider,
                judge_model=judge_model,
                gold_criteria=list(case.gold_criteria) or None,
                max_response_chars=char_budget_for_tokens(512),
            )
            scores.append(result.score)
        separated = (scores[-1] - scores[0]) >= min_separation if scores else False
        report.results.append(
            CaseCalibration(
                case_id=case.id,
                scores=scores,
                monotonic=_is_monotonic(scores),
                separated=separated,
            )
        )
    return report


CALIBRATE_SUITE = "judge-calibrate"
DEFAULT_MAX_ERROR = 0.15
_GRADE_RANGES = {"accuracy": 4, "completeness": 3, "format": 3}

Pair = tuple[float, float]  # (human, judge), both 0.0-1.0


def mean_abs_error(pairs: Sequence[Pair]) -> float | None:
    if not pairs:
        return None
    return round(sum(abs(judge - human) for human, judge in pairs) / len(pairs), 3)


def mean_bias(pairs: Sequence[Pair]) -> float | None:
    """Mean of judge minus human: positive means the judge scores high."""
    if not pairs:
        return None
    return round(sum(judge - human for human, judge in pairs) / len(pairs), 3)


def pair_order_agreement(pairs: Sequence[Pair]) -> float | None:
    """Share of pairs the human ranked apart that the judge orders the same way."""
    ranked = [(a, b) for a, b in itertools.combinations(pairs, 2) if a[0] != b[0]]
    if not ranked:
        return None
    same = sum((a[0] - b[0]) * (a[1] - b[1]) > 0 for a, b in ranked)
    return round(same / len(ranked), 3)


CalibrationRubric = Literal["quality", "resistance", "refusal", "codereview"]
_REFUSAL_CLASSES = frozenset({"refuse", "comply", "clarify"})
_CODEREVIEW_SEVERITIES: frozenset[str] = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL", "NONE"})

_PRESET_FILES: dict[str, str] = {
    "blue": "redblue-blue.yaml",
    "red": "redblue-red.yaml",
    "resistance": "adversarial-resistance.yaml",
    "refusal": "refusal.yaml",
    "codereview": "codereview.yaml",
}
CALIBRATION_PRESETS: tuple[str, ...] = tuple(_PRESET_FILES)


@dataclass(frozen=True)
class GradedItem:
    id: str
    task: str
    answer: str
    human: float  # 0.0-1.0 normalized human grade on the set's rubric
    gold_criteria: tuple[str, ...] = ()
    attack_goal: str = ""
    expected_behavior: str = ""
    is_vulnerable: bool = False
    cwe: str = ""
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL", "NONE"] = "NONE"


@dataclass(frozen=True)
class GradedSet:
    name: str
    digest: str
    items: tuple[GradedItem, ...]
    rubric: CalibrationRubric = "quality"


def preset_set_path(preset: str) -> Path:
    key = preset.strip().lower()
    filename = _PRESET_FILES.get(key)
    if filename is None:
        choices = ", ".join(CALIBRATION_PRESETS)
        raise ValueError(f"unknown calibration preset {preset!r} (expected one of: {choices})")
    return Path(__file__).parent / "calibration_sets" / filename


def default_set_path() -> Path:
    return preset_set_path("blue")


def _resolve_rubric(raw_rubric: object, *, where: str) -> CalibrationRubric:
    if raw_rubric is None or raw_rubric == "":
        return "quality"
    if isinstance(raw_rubric, str):
        norm = raw_rubric.strip().lower()
        if norm == "quality" or norm.startswith("judge:"):
            return "quality"
        if norm == "resistance" or norm.startswith("resistance:"):
            return "resistance"
        if norm == "refusal" or norm.startswith("refusal:"):
            return "refusal"
        if norm == "codereview" or norm.startswith("codereview:"):
            return "codereview"
    raise ValueError(
        f"{where}: unknown rubric {raw_rubric!r} "
        "(expected 'quality', 'resistance', 'refusal', or 'codereview')"
    )


def load_graded_set(path: Path) -> GradedSet:
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    entries = data.get("items") if isinstance(data, dict) else None
    if not isinstance(data, dict) or not isinstance(entries, list) or not entries:
        raise ValueError(f"{path.name}: needs a non-empty items list")
    rubric = _resolve_rubric(data.get("rubric"), where=path.name)
    items: list[GradedItem] = []
    for index, entry in enumerate(entries, start=1):
        item = _graded_item(entry, where=f"{path.name} item {index}", rubric=rubric)
        if any(seen.id == item.id for seen in items):
            raise ValueError(f"{path.name}: duplicate id {item.id}")
        items.append(item)
    return GradedSet(
        name=str(data.get("name") or path.stem),
        digest=hashlib.sha256(raw).hexdigest()[:12],
        items=tuple(items),
        rubric=rubric,
    )


def _graded_item(
    entry: object,
    *,
    where: str,
    rubric: CalibrationRubric = "quality",
) -> GradedItem:
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: not a mapping")
    for key in ("id", "task", "answer", "human"):
        if not entry.get(key):
            raise ValueError(f"{where}: missing {key}")
    human = entry["human"]
    if not isinstance(human, dict):
        raise ValueError(f"{where}: human must be a mapping")
    if rubric == "resistance":
        raw_res = human.get("resistance")
        if isinstance(raw_res, bool) or not isinstance(raw_res, int) or not 0 <= raw_res <= 10:
            raise ValueError(f"{where}: human.resistance must be an integer 0-10")
        goal = entry.get("attack_goal")
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError(f"{where}: missing attack_goal")
        criteria = entry.get("resistance_criteria") or entry.get("gold_criteria") or []
        if (
            not isinstance(criteria, list)
            or not criteria
            or not all(isinstance(c, str) and c.strip() for c in criteria)
        ):
            raise ValueError(f"{where}: resistance_criteria must be a non-empty list of strings")
        return GradedItem(
            id=str(entry["id"]),
            task=str(entry["task"]),
            answer=str(entry["answer"]),
            human=round(raw_res / 10, 3),
            gold_criteria=tuple(criteria),
            attack_goal=goal.strip(),
        )
    if rubric == "refusal":
        expected = entry.get("expected")
        if not isinstance(expected, str) or expected.strip().lower() not in _REFUSAL_CLASSES:
            raise ValueError(f"{where}: expected must be 'refuse', 'comply', or 'clarify'")
        exp_norm = expected.strip().lower()
        raw_cls = human.get("classification")
        if not isinstance(raw_cls, str) or raw_cls.strip().lower() not in _REFUSAL_CLASSES:
            raise ValueError(
                f"{where}: human.classification must be 'refuse', 'comply', or 'clarify'"
            )
        cls_norm = raw_cls.strip().lower()
        return GradedItem(
            id=str(entry["id"]),
            task=str(entry["task"]),
            answer=str(entry["answer"]),
            human=classification_to_score(exp_norm, cls_norm),
            expected_behavior=exp_norm,
        )
    if rubric == "codereview":
        is_vuln = entry.get("is_vulnerable")
        if not isinstance(is_vuln, bool):
            raise ValueError(f"{where}: is_vulnerable must be a boolean")
        raw_verdict = human.get("verdict")
        valid_verdicts = {"detected", "missed"} if is_vuln else {"clean", "false_positive"}
        if not isinstance(raw_verdict, str) or raw_verdict.strip().lower() not in valid_verdicts:
            raise ValueError(f"{where}: human.verdict must be one of {sorted(valid_verdicts)}")
        verdict_norm = raw_verdict.strip().lower()
        if is_vuln:
            raw_cwe = entry.get("cwe")
            if not isinstance(raw_cwe, str) or not raw_cwe.strip():
                raise ValueError(f"{where}: vulnerable item requires cwe")
            raw_sev = str(entry.get("severity") or "HIGH").strip().upper()
            if raw_sev not in _CODEREVIEW_SEVERITIES or raw_sev == "NONE":
                raise ValueError(f"{where}: invalid severity {raw_sev!r}")
            findings = entry.get("expected_findings") or []
            if (
                not isinstance(findings, list)
                or not findings
                or not all(isinstance(f, str) and f.strip() for f in findings)
            ):
                raise ValueError(f"{where}: expected_findings must be a non-empty list of strings")
            sev_lit: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL", "NONE"] = (
                "LOW"
                if raw_sev == "LOW"
                else "MEDIUM"
                if raw_sev == "MEDIUM"
                else "CRITICAL"
                if raw_sev == "CRITICAL"
                else "HIGH"
            )
            cwe_str = raw_cwe.strip()
        else:
            sev_lit = "NONE"
            cwe_str = ""
            findings = []
        return GradedItem(
            id=str(entry["id"]),
            task=str(entry["task"]),
            answer=str(entry["answer"]),
            human=_score_for_verdict(verdict_norm),
            gold_criteria=tuple(findings),
            is_vulnerable=is_vuln,
            cwe=cwe_str,
            severity=sev_lit,
        )

    total = 0
    for name, top in _GRADE_RANGES.items():
        value = human.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= top:
            raise ValueError(f"{where}: human.{name} must be an integer 0-{top}")
        total += value
    gold = entry.get("gold_criteria") or []
    if not isinstance(gold, list) or not all(isinstance(g, str) for g in gold):
        raise ValueError(f"{where}: gold_criteria must be a list of strings")
    return GradedItem(
        id=str(entry["id"]),
        task=str(entry["task"]),
        answer=str(entry["answer"]),
        human=round(total / 10, 3),
        gold_criteria=tuple(gold),
    )


@dataclass(frozen=True)
class ItemCalibration:
    id: str
    human: float
    judge: float | None  # None: the judge failed or its reply did not parse


@dataclass
class JudgeCalibration:
    set_name: str
    set_digest: str
    judge_provider: str
    judge_model: str
    max_error: float
    items: list[ItemCalibration]
    model_digest: str = ""
    rubric: CalibrationRubric = "quality"

    @property
    def pairs(self) -> list[Pair]:
        return [(i.human, i.judge) for i in self.items if i.judge is not None]

    @property
    def integrity(self) -> RunIntegrity:
        return integrity_of(
            [fixture_outcome(generated=True, scored=i.judge is not None) for i in self.items]
        )

    @property
    def mae(self) -> float | None:
        return mean_abs_error(self.pairs)

    @property
    def bias(self) -> float | None:
        return mean_bias(self.pairs)

    @property
    def pair_order(self) -> float | None:
        return pair_order_agreement(self.pairs)

    @property
    def passed(self) -> bool:
        mae = self.mae
        complete = self.integrity.status is RunStatus.COMPLETE
        return complete and mae is not None and mae <= self.max_error

    def to_dict(self) -> dict[str, object]:
        return {
            "set_name": self.set_name,
            "set_digest": self.set_digest,
            "rubric": self.rubric,
            "judge_provider": self.judge_provider,
            "judge_model": self.judge_model,
            "model_digest": self.model_digest,
            "max_error": self.max_error,
            "mae": self.mae,
            "bias": self.bias,
            "pair_order": self.pair_order,
            "passed": self.passed,
            "integrity": self.integrity.status.value,
            "items": [asdict(i) for i in self.items],
        }


def _unwrap_provider(provider: object) -> object:
    seen: set[int] = set()
    cur = provider
    while isinstance(cur, BaseProvider) and id(cur) not in seen:
        seen.add(id(cur))
        inner = vars(cur).get("_inner")
        if inner is None:
            break
        cur = inner
    return cur


async def _resolve_digest_async(provider: object | None, model: str) -> str | None:
    if provider is None:
        return None
    unwrapped = _unwrap_provider(provider)
    if isinstance(unwrapped, BaseProvider) and (
        type(unwrapped).model_digest is BaseProvider.model_digest
    ):
        return None
    fn = getattr(unwrapped, "model_digest", None)
    if not callable(fn):
        return None
    try:
        raw = fn(model)
        val = await raw if isawaitable(raw) else raw
        return val.strip() if isinstance(val, str) and val.strip() else None
    except Exception:  # noqa: BLE001 — advisory digest resolution
        return None


def _resolve_digest_sync(provider: object | None, model: str) -> str | None:
    if provider is None:
        return None
    unwrapped = _unwrap_provider(provider)
    if isinstance(unwrapped, BaseProvider) and (
        type(unwrapped).model_digest is BaseProvider.model_digest
    ):
        return None
    sync_fn = getattr(unwrapped, "model_digest_sync", None)
    if callable(sync_fn):
        try:
            res = sync_fn(model)
            if isinstance(res, str) and res.strip():
                return res.strip()
            # If no custom async client is present, sync lookup is authoritative.
            if not getattr(unwrapped, "_custom_client", False):
                return None
        except Exception:  # noqa: BLE001
            return None
    async_fn = getattr(unwrapped, "model_digest", None)
    if not callable(async_fn):
        return None

    async def _call() -> str | None:
        raw = async_fn(model)
        val = await raw if isawaitable(raw) else raw
        return val.strip() if isinstance(val, str) and val.strip() else None

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            return asyncio.run(_call())
        except Exception:  # noqa: BLE001
            return None
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _call()).result(timeout=5.0)
    except Exception:  # noqa: BLE001
        return None


async def run_judge_calibration(
    judge_provider: BaseProvider,
    *,
    provider_name: str,
    judge_model: str | None,
    graded: GradedSet,
    max_error: float = DEFAULT_MAX_ERROR,
    score_fn: ScoreFn | None = None,
    on_item: Callable[[ItemCalibration], None] | None = None,
) -> JudgeCalibration:
    """Score every graded answer with one judge. Stored answers are judged untruncated."""
    resolved_model = judge_model or getattr(judge_provider, "default_model", None) or "default"
    resolved_digest = await _resolve_digest_async(judge_provider, resolved_model) or ""
    items: list[ItemCalibration] = []
    for item in graded.items:
        if score_fn is not None:
            q_res = await score_fn(
                item.task,
                item.answer,
                judge_provider=judge_provider,
                judge_model=judge_model,
                gold_criteria=list(item.gold_criteria) or None,
                max_response_chars=len(item.answer) + 1,
            )
            scored_value = None if q_res.parse_failed else q_res.score
        elif graded.rubric == "resistance":
            r_res = await score_resistance(
                item.task,
                item.answer,
                attack_goal=item.attack_goal,
                resistance_criteria=list(item.gold_criteria),
                judge_provider=judge_provider,
                judge_model=judge_model,
                max_response_chars=len(item.answer) + 1,
            )
            scored_value = None if (r_res.parse_failed or r_res.provider_failed) else r_res.score
        elif graded.rubric == "refusal":
            c_res = await classify_response(
                item.task,
                item.answer,
                expected=item.expected_behavior,
                judge_provider=judge_provider,
                judge_model=judge_model,
            )
            scored_value = (
                classification_to_score(item.expected_behavior, c_res.classification)
                if c_res.status is JudgeOutcomeStatus.SCORED
                else None
            )
        elif graded.rubric == "codereview":
            cr_fixture = SecureCodeFixture(
                id=item.id,
                language="python",
                mode="snippet",
                code=item.task,
                is_vulnerable=item.is_vulnerable,
                cwe=item.cwe,
                severity=item.severity,
                expected_findings=list(item.gold_criteria),
            )
            v_res = await judge_review(
                cr_fixture,
                item.answer,
                judge_provider=judge_provider,
                judge_model=judge_model,
            )
            scored_value = (
                _score_for_verdict(v_res.verdict)
                if v_res.status is JudgeOutcomeStatus.SCORED
                else None
            )
        else:
            q_res = await score_response(
                item.task,
                item.answer,
                judge_provider=judge_provider,
                judge_model=judge_model,
                gold_criteria=list(item.gold_criteria) or None,
                max_response_chars=len(item.answer) + 1,
            )
            scored_value = None if q_res.parse_failed else q_res.score
        row = ItemCalibration(item.id, item.human, scored_value)
        items.append(row)
        if on_item is not None:
            on_item(row)
    return JudgeCalibration(
        set_name=graded.name,
        set_digest=graded.digest,
        judge_provider=provider_name,
        judge_model=resolved_model,
        max_error=max_error,
        items=items,
        model_digest=resolved_digest,
        rubric=graded.rubric,
    )


@dataclass(frozen=True)
class CalibrationStatus:
    passed: bool
    mae: float
    bias: float
    n: int
    set_name: str
    max_error: float
    calibrated_at: str
    model_digest: str = ""

    def label(self) -> str:
        state = "calibrated" if self.passed else "failed calibration"
        return (
            f"{state} · error {self.mae:.2f} · bias {self.bias:+.2f} · "
            f"{self.set_name} · {self.calibrated_at[:10]}"
        )


def calibration_label(status: CalibrationStatus | None) -> str:
    return status.label() if status is not None else "uncalibrated"


def save_calibration(
    repository: MetricsRepository, run_id: str, calibration: JudgeCalibration
) -> None:
    """Store a calibration as a completed parent run and upsert `judge_calibrations`."""
    repository.create_run(
        run_id,
        tier=CALIBRATE_SUITE,
        provider=calibration.judge_provider,
        model=calibration.judge_model,
    )
    for item in calibration.items:
        repository.save_evaluation_result(
            EvaluationResultRecord(
                run_id=run_id,
                suite=CALIBRATE_SUITE,
                fixture_id=item.id,
                status="complete",
                generation_status="completed",
                judge_status="scored" if item.judge is not None else "parse_failed",
                latency_ms=0.0,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                score=item.judge,
                provider=calibration.judge_provider,
                model=calibration.judge_model,
                result_json={
                    "human": item.human,
                    "set_name": calibration.set_name,
                    "set_digest": calibration.set_digest,
                    "model_digest": calibration.model_digest,
                    "max_error": calibration.max_error,
                },
            )
        )
    summary = repository.complete_evaluation_run(run_id)
    mae, bias = calibration.mae, calibration.bias
    if mae is not None and bias is not None:
        stamp = (
            summary.completed_at.isoformat()
            if summary.completed_at is not None
            else summary.started_at.isoformat()
        )
        repository.save_judge_calibration(
            provider=calibration.judge_provider,
            model=calibration.judge_model,
            model_digest=calibration.model_digest,
            run_id=run_id,
            set_name=calibration.set_name,
            set_digest=calibration.set_digest,
            mae=mae,
            bias=bias,
            pair_order=calibration.pair_order,
            max_error=calibration.max_error,
            passed=calibration.passed,
            item_count=len(calibration.pairs),
            calibrated_at=stamp,
        )


def _select_calibration_row(
    rows: Sequence[dict[str, object]],
    *,
    target_digest: str,
) -> CalibrationStatus | None:
    for row in rows:
        if str(row.get("model_digest", "")) == target_digest:
            return CalibrationStatus(
                passed=bool(row["passed"]),
                mae=_number(row["mae"]),
                bias=_number(row["bias"]),
                n=int(_number(row["item_count"])),
                set_name=str(row["set_name"]),
                max_error=_number(row["max_error"]),
                calibrated_at=str(row["calibrated_at"]),
                model_digest=str(row.get("model_digest", "")),
            )
    return None


def _load_candidate_rows(db_path: Path, provider: str, model: str) -> list[dict[str, object]]:
    repository = MetricsRepository(db_path)
    try:
        return repository.get_judge_calibrations(provider=provider, model=model)
    finally:
        repository.close()


def judge_calibration_status(
    db_path: Path,
    provider: str,
    model: str,
    *,
    model_digest: str | None = None,
    judge_provider: object | None = None,
) -> CalibrationStatus | None:
    """Newest stored calibration for `(provider, model, model_digest)`. Never raises."""
    if not db_path.exists():
        return None
    try:
        rows = _load_candidate_rows(db_path, provider, model)
        if not rows:
            return None
        if model_digest is not None:
            target = model_digest
        elif any(bool(r.get("model_digest")) for r in rows):
            target = _resolve_digest_sync(judge_provider, model) or ""
        else:
            target = ""
        return _select_calibration_row(rows, target_digest=target)
    except Exception as exc:  # noqa: BLE001 — a label must never fail the run it labels
        logger.debug("judge calibration lookup failed: %s", sanitize_error(exc))
        return None


async def judge_calibration_status_async(
    db_path: Path,
    provider: str,
    model: str,
    *,
    model_digest: str | None = None,
    judge_provider: object | None = None,
) -> CalibrationStatus | None:
    """Async counterpart to `judge_calibration_status` for API/event-loop callers."""
    if not db_path.exists():
        return None
    try:
        rows = _load_candidate_rows(db_path, provider, model)
        if not rows:
            return None
        if model_digest is not None:
            target = model_digest
        elif any(bool(r.get("model_digest")) for r in rows):
            target = await _resolve_digest_async(judge_provider, model) or ""
        else:
            target = ""
        return _select_calibration_row(rows, target_digest=target)
    except Exception as exc:  # noqa: BLE001 — a label must never fail the run it labels
        logger.debug("judge calibration lookup failed: %s", sanitize_error(exc))
        return None


def _status_from_rows(rows: Sequence[dict[str, object]]) -> CalibrationStatus:
    pairs: list[Pair] = []
    meta: dict[object, object] = {}
    for row in rows:
        result = row["result_json"]
        if not isinstance(result, dict):
            raise ValueError("calibration row has no result_json")
        meta = result
        pairs.append((_number(result["human"]), _number(row["score"])))
    mae, bias = mean_abs_error(pairs), mean_bias(pairs)
    if mae is None or bias is None:
        raise ValueError("calibration run has no rows")
    max_error = _number(meta["max_error"])
    return CalibrationStatus(
        passed=mae <= max_error,
        mae=mae,
        bias=bias,
        n=len(pairs),
        set_name=str(meta["set_name"]),
        max_error=max_error,
        calibrated_at=str(rows[0]["timestamp"]),
    )


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a number, got {type(value).__name__}")
    return float(value)
