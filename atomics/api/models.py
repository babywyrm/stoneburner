"""Pydantic models for the atomics API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Upper bounds exist so one authenticated request cannot pin the server or run
# up an unbounded provider bill. They are deliberately generous: the CLI is the
# path for genuinely long campaigns and is not affected by these.
MAX_ITERATIONS = 1000
MAX_INTERVAL_SECONDS = 3600
MAX_FIXTURES = 500
MAX_TREND_HOURS = 168

# Dollar ceiling for one API-triggered eval, shared by the model and its judge.
# Eval suites are the only remotely reachable path that spends without a tier
# profile behind it, so unlike the CLI this is always applied.
DEFAULT_EVAL_BUDGET_USD = 10.0
MAX_EVAL_BUDGET_USD = 1000.0

# A sweep is models × suites × runs. The CLI can do a 32-model overnight;
# a remote caller cannot. Eight models, the six known suites, three passes
# is the most one API request may start.
MAX_SWEEP_MODELS = 8
MAX_SWEEP_RUNS = 3
# CLI --runs 3 exists on these suites. Accuracy / multiturn / rag / … are one pass.
EVAL_RUNS_SUITES = frozenset({"adversarial", "redblue", "refusal", "codereview", "toolcall"})
# Suites whose runners pin the model's sampling temperature (CLI --temperature).
EVAL_TEMPERATURE_SUITES = frozenset(
    {"accuracy", "adversarial", "redblue", "refusal", "codereview", "toolcall"}
)
# Suites with CLI --category / --mode subsetting.
EVAL_CATEGORY_SUITES = frozenset({"adversarial", "toolcall", "redblue"})
EVAL_CHANNELS = frozenset({"both", "tools", "prose"})

# Load tests. The CLI can soak for hours at c=8; a remote caller cannot.
MAX_STRESS_CONCURRENCY = 8
MAX_STRESS_PHASE_SECONDS = 15.0
MAX_SOAK_DURATION_SECONDS = 300
MAX_SOAK_CONCURRENCY = 4
MAX_LOAD_PREDICT = 256


def _normalize_effort_field(value: str | None) -> str | None:
    from atomics.providers.effort import EffortError, normalize_effort

    try:
        return normalize_effort(value)
    except EffortError as exc:
        raise ValueError(str(exc)) from exc


def _normalize_reasoning_mode_field(value: str | None) -> str | None:
    from atomics.providers.effort import EffortError, normalize_reasoning_mode

    try:
        return normalize_reasoning_mode(value)
    except EffortError as exc:
        raise ValueError(str(exc)) from exc


class RunRequest(BaseModel):
    """Request body to start a benchmark run."""

    provider: str
    model: str | None = None
    tier: str = "ez"
    iterations: int = Field(default=3, ge=1, le=MAX_ITERATIONS)
    interval: int = Field(default=5, ge=0, le=MAX_INTERVAL_SECONDS)
    # Accepted for old clients and ignored: a benchmark run always persists.
    save: bool = True
    thinking: bool | None = None
    effort: str | None = None
    reasoning_mode: str | None = None
    host: str | None = None

    @field_validator("effort")
    @classmethod
    def _known_effort(cls, value: str | None) -> str | None:
        return _normalize_effort_field(value)

    @field_validator("reasoning_mode")
    @classmethod
    def _known_reasoning_mode(cls, value: str | None) -> str | None:
        return _normalize_reasoning_mode_field(value)


def category_catalog_for_suite(suite: str, parts: list[str]) -> tuple[list[Any], set[str]]:
    """Return ``(matching_fixtures, valid_category_tokens)`` for a suite."""
    if suite == "adversarial":
        from atomics.eval.adversarial import ALL_FIXTURES as ADV_ALL
        from atomics.eval.adversarial import GROUP_ALIASES as ADV_ALIASES
        from atomics.eval.adversarial import select_fixtures as select_adversarial

        valid = set(ADV_ALIASES) | {f.category for f in ADV_ALL}
        return list(select_adversarial(parts)), valid
    if suite == "toolcall":
        from atomics.eval.toolcall.fixtures import (
            ALL_FIXTURES as TC_ALL,
        )
        from atomics.eval.toolcall.fixtures import (
            GROUP_ALIASES as TC_ALIASES,
        )
        from atomics.eval.toolcall.fixtures import (
            fixtures_for_category,
        )

        valid = set(TC_ALIASES) | {f.category for f in TC_ALL}
        return list(fixtures_for_category(parts)), valid
    if suite == "redblue":
        from atomics.eval.redblue.fixtures import ALL_FIXTURES as RB_ALL

        valid = {"red", "blue", "all"} | {f.category for f in RB_ALL}
        wanted = set(parts)
        if "all" in wanted:
            return list(RB_ALL), valid
        return [f for f in RB_ALL if f.team in wanted or f.category in wanted], valid
    return [], set()


class EvalRequest(BaseModel):
    """Request body to start an eval suite."""

    model_config = ConfigDict(extra="forbid")

    suite: str
    provider: str
    model: str | None = None
    judge_model: str | None = None
    host: str | None = None
    fixtures: list[str] | None = Field(default=None, max_length=MAX_FIXTURES)
    category: str | None = None
    # Accepted for old clients and ignored: API evals return results and never persist.
    save: bool = True
    thinking: bool | None = None
    effort: str | None = None
    reasoning_mode: str | None = None
    budget_usd: float = Field(default=DEFAULT_EVAL_BUDGET_USD, gt=0, le=MAX_EVAL_BUDGET_USD)
    judge_host: str | None = None
    runs: int = Field(default=1, ge=1, le=MAX_SWEEP_RUNS)
    channel: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    profile: str | None = None

    @field_validator("profile")
    @classmethod
    def _profile_under_profiles(cls, value: str | None) -> str | None:
        if value is None:
            return None
        root = Path("profiles").resolve()
        path = Path(value)
        if (
            path.is_absolute()
            or path.suffix.lower() not in {".yaml", ".yml"}
            or not path.resolve().is_relative_to(root)
        ):
            raise ValueError("profile must be a relative .yaml/.yml path under profiles/")
        return value

    @field_validator("effort")
    @classmethod
    def _known_effort(cls, value: str | None) -> str | None:
        return _normalize_effort_field(value)

    @field_validator("reasoning_mode")
    @classmethod
    def _known_reasoning_mode(cls, value: str | None) -> str | None:
        return _normalize_reasoning_mode_field(value)

    @model_validator(mode="after")
    def _runs_and_channel_only_on_matching_suites(self) -> EvalRequest:
        if self.category is not None:
            suite_norm = self.suite.strip().lower()
            if suite_norm not in EVAL_CATEGORY_SUITES:
                raise ValueError(
                    f"category is only valid for {sorted(EVAL_CATEGORY_SUITES)}; "
                    f"{self.suite} does not support category filtering"
                )
            parts = [part.strip() for part in self.category.split(",") if part.strip()]
            if not parts:
                raise ValueError("category must not be empty")
            catalog, valid = category_catalog_for_suite(suite_norm, parts)
            unknown = [part for part in parts if part not in valid]
            if unknown:
                raise ValueError(
                    f"unknown {suite_norm} category: {', '.join(unknown)}. "
                    f"Valid: {', '.join(sorted(valid))}"
                )
            if self.fixtures is not None:
                allowed_ids = {f.id for f in catalog}
                outside = [fid for fid in self.fixtures if fid not in allowed_ids]
                if outside:
                    raise ValueError(
                        f"fixture IDs outside category {','.join(parts)!r}: {', '.join(outside)}"
                    )
            self.category = ",".join(parts)
        if self.profile is not None:
            from atomics.eval.batteries import PROFILE_SUITES

            if self.suite not in PROFILE_SUITES:
                raise ValueError(
                    f"profile is only valid for {sorted(PROFILE_SUITES)}; "
                    f"{self.suite} does not support profile targets"
                )
        if self.runs > 1 and self.suite not in EVAL_RUNS_SUITES:
            raise ValueError(
                f"runs is only valid for {sorted(EVAL_RUNS_SUITES)}; "
                f"{self.suite} is a single-pass suite"
            )
        if self.temperature is not None and self.suite not in EVAL_TEMPERATURE_SUITES:
            raise ValueError(
                f"temperature is only valid for {sorted(EVAL_TEMPERATURE_SUITES)}; "
                f"{self.suite} does not pin it"
            )
        if self.channel is None:
            return self
        channel = self.channel.strip().lower()
        if channel not in EVAL_CHANNELS:
            raise ValueError(f"channel must be one of {sorted(EVAL_CHANNELS)}")
        if self.suite != "toolcall":
            raise ValueError("channel is only valid for toolcall")
        self.channel = channel
        return self


class SweepRequest(BaseModel):
    """Start a multi-model, multi-suite campaign.

    Budget is required: unlike a single eval there is no default ceiling,
    because a forgotten default of $10 against eight models is not a ceiling,
    it is a surprise stop halfway through. Name the models; there is no
    discover-everything flag.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str
    models: list[str] = Field(min_length=1, max_length=MAX_SWEEP_MODELS)
    suites: list[str] = Field(min_length=1)
    judge_model: str | None = None
    runs: int = Field(default=1, ge=1, le=MAX_SWEEP_RUNS)
    thinking: bool | None = None
    effort: str | None = None
    reasoning_mode: str | None = None
    budget_usd: float = Field(gt=0, le=MAX_EVAL_BUDGET_USD)
    host: str | None = None
    judge_host: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)

    @field_validator("effort")
    @classmethod
    def _known_effort(cls, value: str | None) -> str | None:
        return _normalize_effort_field(value)

    @field_validator("reasoning_mode")
    @classmethod
    def _known_reasoning_mode(cls, value: str | None) -> str | None:
        return _normalize_reasoning_mode_field(value)

    @field_validator("models")
    @classmethod
    def _clean_models(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError("models must not be empty")
        return cleaned

    @field_validator("suites")
    @classmethod
    def _known_suites(cls, value: list[str]) -> list[str]:
        from atomics.eval.gauntlet import parse_suites

        return parse_suites(",".join(value))


class BatteryRequest(BaseModel):
    """Run a named battery as one job. Budget is required (paid judge possible)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    provider: str
    model: str | None = None
    budget_usd: float = Field(gt=0, le=MAX_EVAL_BUDGET_USD)
    judge_model: str | None = None
    judge_host: str | None = None
    host: str | None = None
    # CLI `battery` sends --no-thinking unless asked, so short fixtures stay visible.
    thinking: bool = False
    effort: str | None = None
    reasoning_mode: str | None = None
    # None keeps each step's own default; runs apply only to eval.batteries.RUNS_SUITES steps.
    runs: int | None = Field(default=None, ge=1, le=MAX_SWEEP_RUNS)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    profile: str | None = None

    @field_validator("thinking", mode="before")
    @classmethod
    def _null_thinking_is_the_default(cls, value: object) -> object:
        return False if value is None else value

    @field_validator("profile")
    @classmethod
    def _profile_under_profiles(cls, value: str | None) -> str | None:
        if value is None:
            return None
        root = Path("profiles").resolve()
        path = Path(value)
        if (
            path.is_absolute()
            or path.suffix.lower() not in {".yaml", ".yml"}
            or not path.resolve().is_relative_to(root)
        ):
            raise ValueError("profile must be a relative .yaml/.yml path under profiles/")
        return value

    @field_validator("effort")
    @classmethod
    def _known_effort(cls, value: str | None) -> str | None:
        return _normalize_effort_field(value)

    @field_validator("reasoning_mode")
    @classmethod
    def _known_reasoning_mode(cls, value: str | None) -> str | None:
        return _normalize_reasoning_mode_field(value)

    @field_validator("name")
    @classmethod
    def _known_battery(cls, value: str) -> str:
        from atomics.eval.batteries import get_battery

        try:
            get_battery(value)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        return value


class StressRequest(BaseModel):
    """Ramp concurrency to find saturation. One named model, required budget."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    budget_usd: float = Field(gt=0, le=MAX_EVAL_BUDGET_USD)
    max_concurrency: int = Field(default=4, ge=1, le=MAX_STRESS_CONCURRENCY)
    phase_seconds: float = Field(default=10.0, gt=0, le=MAX_STRESS_PHASE_SECONDS)
    host: str | None = None

    @field_validator("model")
    @classmethod
    def _clean_model(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("model must not be empty")
        return cleaned


class SoakRequest(BaseModel):
    """Hold fixed concurrency and classify drift. Duration is seconds, max 300."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    budget_usd: float = Field(gt=0, le=MAX_EVAL_BUDGET_USD)
    duration_seconds: int = Field(default=60, ge=30, le=MAX_SOAK_DURATION_SECONDS)
    concurrency: int = Field(default=2, ge=1, le=MAX_SOAK_CONCURRENCY)
    sample_interval: int = Field(default=15, ge=10, le=60)
    host: str | None = None

    @field_validator("model")
    @classmethod
    def _clean_model(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("model must not be empty")
        return cleaned

    @model_validator(mode="after")
    def _interval_fits_duration(self) -> SoakRequest:
        if self.sample_interval >= self.duration_seconds:
            raise ValueError("sample_interval must be less than duration_seconds")
        return self


class JobResponse(BaseModel):
    """Response representing a job."""

    job_id: str
    status: str
    kind: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error: dict[str, object] | None = None
    result_url: str | None = None
    result: Any | None = None
    request: dict[str, object] | None = None
    progress: dict[str, object] | None = None


class ErrorResponse(BaseModel):
    """Error response."""

    detail: str


class HealthResponse(BaseModel):
    """Liveness response: this process is running and serving requests.

    Deliberately checks nothing external. A liveness probe answers "should this
    process be restarted", and restarting the API server does not repair an
    unreachable database — it just removes the endpoint that could have told
    you what was wrong. Dependency checks belong on `/ready`.
    """

    status: str


class ReadinessCheck(BaseModel):
    """One dependency's contribution to readiness."""

    name: str
    ok: bool
    detail: str | None = None


class ReadinessResponse(BaseModel):
    """Readiness: whether this server can currently serve real work."""

    status: str
    checks: list[ReadinessCheck]


class CompareResponse(BaseModel):
    by: str
    rows: list[dict[str, object]]


class ReportResponse(BaseModel):
    runs: list[dict[str, object]]


class TrendsResponse(BaseModel):
    hours: int
    rows: list[dict[str, object]]


class JobSummary(BaseModel):
    """In-memory job without the result payload."""

    job_id: str
    status: str
    kind: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error: dict[str, object] | None = None
    request: dict[str, object] | None = None


class JobsListResponse(BaseModel):
    jobs: list[JobSummary]


class ModelsResponse(BaseModel):
    provider: str
    models: list[dict[str, object]]


class ProviderTestRequest(BaseModel):
    """Probe a provider. The generate prompt is fixed server-side."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str | None = None
    host: str | None = None
    thinking: bool | None = None
    effort: str | None = None
    reasoning_mode: str | None = None

    @field_validator("effort")
    @classmethod
    def _known_effort(cls, value: str | None) -> str | None:
        return _normalize_effort_field(value)

    @field_validator("reasoning_mode")
    @classmethod
    def _known_reasoning_mode(cls, value: str | None) -> str | None:
        return _normalize_reasoning_mode_field(value)


class ProviderTestResponse(BaseModel):
    ok: bool
    health: bool
    provider: str
    model: str | None = None
    response: str | None = None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    thinking_tokens: int = 0
    think: bool = False
    latency_ms: float = 0.0
    cost_usd: float = 0.0
