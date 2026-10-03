"""What a model host declares about its models, and what a probe found.

Each capability resolves from the strongest source present: a live probe,
then the host's declaration, then our name table (thinking only). Tools are
never read from a name. Sources that disagree are listed, not hidden.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal

from atomics.benchmark.model_classes import classify_model, supports_thinking

SCHEMA = 1
CAPABILITIES = ("completion", "tools", "thinking", "vision", "audio")

Source = Literal["probe", "declared", "name-table", "unknown"]


@dataclass(frozen=True)
class Capability:
    value: bool | None
    source: Source


@dataclass(frozen=True)
class Reply:
    answered: bool
    outcome: str
    output_tokens: int
    thinking_tokens: int
    latency_ms: float
    tokens_per_second: float | None


Verdict = Literal["off-works", "off-ignored", "inline", "no-channel"]


@dataclass
class ProbeResult:
    off: Reply | None = None
    on: Reply | None = None
    verdict: Verdict | None = None

    @property
    def recommended(self) -> str | None:
        """The thinking flag to evaluate this model with."""
        if self.verdict is None:
            return None
        return "--thinking" if self.verdict == "off-ignored" else "--no-thinking"

    def to_dict(self) -> dict[str, object]:
        return {
            "off": asdict(self.off) if self.off else None,
            "on": asdict(self.on) if self.on else None,
            "verdict": self.verdict,
            "recommended": self.recommended,
        }


@dataclass
class ModelRecord:
    host: str
    name: str
    digest: str | None = None
    size_bytes: int | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    declared_context: int | None = None
    requested_context: int | None = None
    loaded_context: int | None = None
    # None means the host declares nothing, not that it declares no capabilities.
    declared: frozenset[str] | None = None
    probed: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    probe: ProbeResult | None = None
    marks: list[str] = field(default_factory=list)

    @property
    def model_class(self) -> str:
        return classify_model(self.name).value

    def _sources(self, cap: str) -> dict[Source, bool]:
        seen: dict[Source, bool] = {}
        if cap in self.probed:
            seen["probe"] = self.probed[cap]
        if self.declared is not None:
            seen["declared"] = cap in self.declared
        if cap == "thinking":
            seen["name-table"] = supports_thinking(self.name)
        return seen

    def claims(self, cap: str) -> bool:
        """Any declaration or name table says yes, whatever a probe found."""
        return any(v for s, v in self._sources(cap).items() if s != "probe")

    def capability(self, cap: str) -> Capability:
        seen = self._sources(cap)
        if not seen:
            return Capability(None, "unknown")
        source, value = next(iter(seen.items()))
        return Capability(value, source)

    @property
    def disagreements(self) -> list[str]:
        out: list[str] = []
        for cap in CAPABILITIES:
            seen = self._sources(cap)
            if len(set(seen.values())) > 1:
                pairs = " ".join(f"{s}={str(v).lower()}" for s, v in seen.items())
                out.append(f"{cap}: {pairs}")
        return out

    @property
    def flags(self) -> list[str]:
        out: list[str] = []
        req = self.requested_context
        if req and self.declared_context and req > self.declared_context:
            out.append("requested-above-declared")
        if req and self.loaded_context and self.loaded_context != req:
            out.append("loaded-at-other-context")
        return out + self.marks

    @property
    def evaluable(self) -> bool:
        return self.capability("completion").value is not False

    def to_dict(self) -> dict[str, object]:
        caps = {c: self.capability(c) for c in CAPABILITIES}
        return {
            "host": self.host,
            "name": self.name,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
            "parameter_size": self.parameter_size,
            "quantization": self.quantization,
            "declared_context": self.declared_context,
            "requested_context": self.requested_context,
            "loaded_context": self.loaded_context,
            "flags": self.flags,
            "model_class": self.model_class,
            "capabilities": {c: {"value": v.value, "source": v.source} for c, v in caps.items()},
            "evaluable": self.evaluable,
            "disagreements": self.disagreements,
            "errors": list(self.errors),
            "probe": self.probe.to_dict() if self.probe else None,
        }


@dataclass
class HostRecord:
    label: str
    provider: str
    version: str | None = None
    error: str | None = None
    loaded: list[tuple[str, int | None]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "provider": self.provider,
            "version": self.version,
            "error": self.error,
            "loaded": [{"name": n, "context": c} for n, c in self.loaded],
        }


@dataclass
class Inventory:
    taken_at: str
    hosts: list[HostRecord]
    models: list[ModelRecord]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "taken_at": self.taken_at,
            "hosts": [h.to_dict() for h in self.hosts],
            "models": [m.to_dict() for m in self.models],
        }


def mark_digest_mismatches(models: Sequence[ModelRecord]) -> None:
    """Flag a tag whose weights differ between hosts: same name, different model."""
    digests: dict[str, set[str]] = {}
    for m in models:
        if m.digest:
            digests.setdefault(m.name, set()).add(m.digest)
    for m in models:
        if len(digests.get(m.name, ())) > 1:
            m.marks.append("digest-mismatch")
