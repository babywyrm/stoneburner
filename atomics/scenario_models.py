"""Data models for mixed-workload scenario testing."""

from __future__ import annotations

from dataclasses import dataclass, field

WORKLOAD_TYPES: frozenset[str] = frozenset({"gate", "eval"})

DEFAULT_NUM_PREDICT: dict[str, int] = {
    "gate": 32,
    "eval": 256,
}

BASELINE_DURATION_SECONDS: float = 5.0


@dataclass
class WorkloadSpec:
    name: str
    type: str
    model: str
    concurrency: int
    sla_ms: float | None = None
    num_predict: int = 0
    prompts_file: str | None = None
    prompts: list[str] = field(default_factory=list)
    profile: str | None = None

    def __post_init__(self) -> None:
        if self.type not in WORKLOAD_TYPES:
            raise ValueError(f"Unknown workload type '{self.type}'. Valid: {sorted(WORKLOAD_TYPES)}")
        if self.concurrency < 1:
            raise ValueError(f"Concurrency must be >= 1, got {self.concurrency}")
        if self.num_predict <= 0:
            self.num_predict = DEFAULT_NUM_PREDICT.get(self.type, 32)


@dataclass
class WorkloadResult:
    spec: WorkloadSpec
    requests: int = 0
    failed: int = 0
    latencies: list[float] = field(default_factory=list)
    per_request_tps: list[float] = field(default_factory=list)
    total_output_tokens: int = 0

    @property
    def p50_ms(self) -> float:
        return _percentile(self.latencies, 50)

    @property
    def p95_ms(self) -> float:
        return _percentile(self.latencies, 95)

    @property
    def avg_tps(self) -> float:
        if not self.per_request_tps:
            return 0.0
        return sum(self.per_request_tps) / len(self.per_request_tps)

    @property
    def sla_violations(self) -> int:
        if self.spec.sla_ms is None:
            return 0
        return sum(1 for lat in self.latencies if lat > self.spec.sla_ms)

    @property
    def sla_compliance_pct(self) -> float:
        if self.spec.sla_ms is None or not self.latencies:
            return 100.0
        return (1 - self.sla_violations / len(self.latencies)) * 100


@dataclass
class ScenarioResult:
    duration_seconds: float = 0.0
    ramp_seconds: float = 0.0
    workloads: list[WorkloadResult] = field(default_factory=list)
    baselines: dict[str, float] = field(default_factory=dict)
    interference: dict[str, float] = field(default_factory=dict)
    total_requests: int = 0
    total_failed: int = 0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100)
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


def parse_workload_flag(flag: str) -> WorkloadSpec:
    """Parse CLI shorthand: type:model:concurrency[:sla_ms]

    Model names can contain colons (e.g. qwen2.5:3b). We parse right-to-left:
    the last 1-2 segments are concurrency and optional sla_ms (both numeric),
    the first segment is the type, and everything in between is the model name.
    """
    parts = flag.split(":")
    if len(parts) < 3:
        raise ValueError(
            f"Invalid workload format '{flag}'. "
            "Expected type:model:concurrency[:sla_ms]"
        )

    wtype = parts[0]

    sla_ms: float | None = None
    # Try parsing the last element as sla_ms and second-to-last as concurrency
    if len(parts) >= 4:
        try:
            maybe_sla = float(parts[-1])
            maybe_conc = int(parts[-2])
            # Both parsed — last is sla, second-to-last is concurrency
            model = ":".join(parts[1:-2])
            concurrency = maybe_conc
            sla_ms = maybe_sla
        except (ValueError, IndexError):
            # Last element is not a valid sla_ms; treat it as concurrency
            try:
                concurrency = int(parts[-1])
            except ValueError:
                raise ValueError(f"Invalid concurrency '{parts[-1]}' in workload '{flag}'")
            model = ":".join(parts[1:-1])
    else:
        # Exactly 3 parts: type:model:concurrency
        try:
            concurrency = int(parts[-1])
        except ValueError:
            raise ValueError(f"Invalid concurrency '{parts[-1]}' in workload '{flag}'")
        model = parts[1]

    if not model:
        raise ValueError(f"Empty model in workload '{flag}'")

    name = f"{wtype}-{model.replace(':', '-')}"

    return WorkloadSpec(
        name=name,
        type=wtype,
        model=model,
        concurrency=concurrency,
        sla_ms=sla_ms,
    )


def load_scenario_yaml(path: str) -> list[WorkloadSpec]:
    """Load workload specs from a YAML scenario file."""
    from pathlib import Path

    import yaml

    content = Path(path).read_text()
    data = yaml.safe_load(content)

    if not isinstance(data, dict) or "workloads" not in data:
        raise ValueError("Scenario file must have a 'workloads' key at the top level")

    specs: list[WorkloadSpec] = []
    for i, entry in enumerate(data["workloads"]):
        name = entry.get("name", f"workload-{i}")
        wtype = entry.get("type", "")
        model = entry.get("model", "")
        concurrency = entry.get("concurrency", 1)
        sla_ms = entry.get("sla_ms")
        num_predict = entry.get("num_predict", 0)
        prompts_file = entry.get("prompts_file")
        profile = entry.get("profile")

        if not model and not profile:
            raise ValueError(f"Workload '{name}' missing required 'model' or 'profile' field")

        specs.append(WorkloadSpec(
            name=name,
            type=wtype,
            model=model,
            concurrency=concurrency,
            sla_ms=sla_ms,
            num_predict=num_predict,
            prompts_file=prompts_file,
            profile=profile,
        ))

    return specs
