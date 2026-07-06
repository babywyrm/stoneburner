"""Reference reader/resolver for the ``inference.env`` standard.

A vendor-neutral control file lets any box describe the LLM inference target it
is wired to, so any consumer (the ``atomics`` providers, the ``brain/`` scripts,
a downstream agent) can self-configure. See ``docs/INFERENCE_ENV.md`` for the
full specification.

This module is intentionally dependency-light (stdlib + ``atomics.providers``)
and carries **no** box-specific knowledge — no hosts, credentials, or k8s glue.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass, field
from typing import Any

# Canonical backend names == atomics provider names.
CANONICAL_BACKENDS: frozenset[str] = frozenset(
    {"ollama", "vllm", "openai", "claude", "bedrock", "brain-gateway"}
)
UNIVERSAL_TIERS: tuple[str, ...] = ("easy", "medium", "hard")

# Search order for load_control_file() when no explicit path is given.
_DEFAULT_PATHS: tuple[str, ...] = (
    "/opt/agentic/inference.env",
    "/etc/agentic/inference.env",
)
_ENV_OVERRIDES: tuple[str, ...] = ("INFERENCE_ENV", "BRAIN_ENV")

_TRUE: frozenset[str] = frozenset({"1", "true", "yes", "on"})


# ── env parsing + legacy normalization ────────────────────────────────────────

def parse_env(text: str) -> dict[str, str]:
    """Parse KEY=VALUE lines; ignore comments and blanks. Values kept verbatim."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        out[k.strip()] = v.strip()
    return out


def normalize_legacy(raw: dict[str, str]) -> dict[str, str]:
    """Fold legacy ``brain/`` keys into the canonical ``INFERENCE_*`` schema.

    Canonical keys always win when both forms are present.
    """
    out = dict(raw)

    # Backend: INFERENCE_BACKEND wins; else map legacy INFERENCE_API.
    if "INFERENCE_BACKEND" not in out:
        api = out.get("INFERENCE_API", "").strip().lower()
        if api == "ollama":
            out["INFERENCE_BACKEND"] = "ollama"
        elif api == "openai":
            # brain/ "openai mode" means a LOCAL OpenAI-compatible gateway.
            out["INFERENCE_BACKEND"] = "vllm"

    backend = out.get("INFERENCE_BACKEND", "").strip().lower()
    # Pick the legacy field family that matches the resolved backend.
    if backend == "ollama":
        _fill(out, "INFERENCE_URL", raw.get("OLLAMA_URL"))
        _fill(out, "INFERENCE_MODEL", raw.get("OLLAMA_MODEL"))
        _fill(out, "INFERENCE_THINK", raw.get("OLLAMA_THINK"))
    elif backend in {"vllm", "openai"}:
        _fill(out, "INFERENCE_URL", raw.get("OPENAI_BASE_URL"))
        _fill(out, "INFERENCE_MODEL", raw.get("OPENAI_MODEL"))
        _fill(out, "INFERENCE_API_KEY", raw.get("OPENAI_API_KEY"))
    return out


def _fill(d: dict[str, str], key: str, value: str | None) -> None:
    if key not in d and value is not None:
        d[key] = value


# ── InferenceTarget ───────────────────────────────────────────────────────────

@dataclass
class InferenceTarget:
    """A resolved inference target read from an ``inference.env`` file."""

    backend: str = ""
    url: str = ""
    model: str = ""
    think: bool = False
    api_key: str = ""
    difficulty: str = ""
    pool: str = ""
    resolved_at: str = ""
    resolved_by: str = ""
    raw: dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def from_mapping(cls, raw: dict[str, str]) -> InferenceTarget:
        norm = normalize_legacy(raw)
        return cls(
            backend=norm.get("INFERENCE_BACKEND", "").strip().lower(),
            url=norm.get("INFERENCE_URL", ""),
            model=norm.get("INFERENCE_MODEL", ""),
            think=norm.get("INFERENCE_THINK", "false").strip().lower() in _TRUE,
            api_key=norm.get("INFERENCE_API_KEY", ""),
            difficulty=norm.get("INFERENCE_DIFFICULTY", ""),
            pool=norm.get("INFERENCE_POOL", ""),
            resolved_at=norm.get("INFERENCE_RESOLVED_AT", ""),
            resolved_by=norm.get("INFERENCE_RESOLVED_BY", ""),
            raw=norm,
        )

    @classmethod
    def from_text(cls, text: str) -> InferenceTarget:
        return cls.from_mapping(parse_env(text))


def load_control_file(path: str | None = None) -> InferenceTarget | None:
    """Load and normalize a control file.

    With ``path``: read that file (returns ``None`` if absent).
    Without: search ``$INFERENCE_ENV``, ``$BRAIN_ENV``, then the default paths;
    returns ``None`` if nothing is found so callers can fall back cleanly.
    """
    candidates: list[str] = []
    if path is not None:
        candidates.append(path)
    else:
        candidates.extend(os.environ[e] for e in _ENV_OVERRIDES if os.environ.get(e))
        candidates.extend(_DEFAULT_PATHS)
    for cand in candidates:
        if cand and os.path.isfile(cand):
            with open(cand) as f:
                return InferenceTarget.from_text(f.read())
    return None


# ── resolver (agnostic intent -> resolved) ────────────────────────────────────

def resolve_model(machine: dict, difficulty: str) -> str:
    dm = machine.get("difficulty_models") or {}
    if difficulty not in dm:
        have = ", ".join(sorted(dm)) or "(none)"
        raise ValueError(f"no model mapped for difficulty '{difficulty}' (have: {have})")
    return dm[difficulty]


def resolve_endpoint(profile: dict) -> dict:
    ep = profile.get("endpoint") or {}
    host = str(ep.get("host", ""))
    port = str(ep.get("port", ""))
    url = ep.get("url") or (f"http://{host}:{port}" if host and port else "")
    return {
        "host": host,
        "port": port,
        "url": url,
        "backend": profile.get("backend", "ollama"),
        "api_key": profile.get("api_key", "") or "",
    }


def check_model_compat(machine: dict, model: str) -> tuple[str, str]:
    compat = machine.get("model_compatibility", {}) or {}
    incompatible: set[str] = set()
    compatible: set[str] = set()
    for key, val in compat.items():
        if not isinstance(val, list):
            continue
        (incompatible if "incompat" in key else compatible).update(val)
    if model in incompatible:
        return ("INCOMPATIBLE", f"{model} is listed incompatible for this machine")
    if compatible and model not in compatible:
        return ("UNTESTED", f"{model} not in compatible list")
    return ("OK", "")


def check_backend(machine: dict, backend: str) -> bool:
    return backend in (machine.get("supported_backends") or ["ollama"])


def render_env(intent: dict, resolved: dict, resolved_by: str) -> str:
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# Inference control file — see docs/INFERENCE_ENV.md",
        "",
        "# INTENT",
        f"INFERENCE_DIFFICULTY={intent['difficulty']}",
        f"INFERENCE_POOL={intent['pool']}",
        "",
        "# RESOLVED",
        f"INFERENCE_BACKEND={resolved['backend']}",
        f"INFERENCE_URL={resolved['url']}",
        f"INFERENCE_MODEL={resolved['model']}",
        f"INFERENCE_THINK={resolved['think']}",
        f"INFERENCE_API_KEY={resolved['api_key']}",
        "",
        "# PROVENANCE",
        f"INFERENCE_RESOLVED_AT={ts}",
        f"INFERENCE_RESOLVED_BY={resolved_by}",
        "",
    ]
    return "\n".join(lines)


def resolve(machine: dict, profile: dict, difficulty: str, pool: str,
            resolved_by: str) -> dict:
    model = resolve_model(machine, difficulty)
    ep = resolve_endpoint(profile)
    think = str(machine.get("think_default", "false")).lower()
    resolved = {**ep, "model": model, "think": think}
    intent = {"difficulty": difficulty, "pool": pool}
    return {
        "intent": intent,
        "resolved": resolved,
        "compat": check_model_compat(machine, model),
        "backend_ok": check_backend(machine, ep["backend"]),
        "env": render_env(intent, resolved, resolved_by),
    }


# ── provider auto-load integration point ──────────────────────────────────────

def provider_from_target(target: InferenceTarget, *, client: Any | None = None):
    """Build the matching ``atomics`` provider for a resolved target.

    Lazy-imports providers so importing this module stays cheap and free of
    optional deps (e.g. the ``openai`` package). ``client`` is backend-specific
    (httpx.AsyncClient for ollama/vllm, AsyncOpenAI for openai), so it is typed
    ``Any`` at this generic dispatch boundary.
    """
    backend = target.backend
    if backend == "ollama":
        from atomics.providers.ollama import OllamaProvider

        url = target.url or "http://localhost:11434"
        return OllamaProvider(host=url, default_model=target.model, client=client)
    if backend == "vllm":
        from atomics.providers.vllm import VllmProvider

        url = target.url or "http://localhost:8000/v1"
        return VllmProvider(base_url=url, default_model=target.model,
                            api_key=target.api_key or "dummy", client=client)
    if backend == "openai":
        from atomics.providers.openai import OpenAIProvider

        return OpenAIProvider(api_key=target.api_key, default_model=target.model,
                              client=client)
    raise ValueError(
        f"unknown/unsupported backend '{backend}' "
        f"(expected one of: {', '.join(sorted(CANONICAL_BACKENDS))})"
    )


def load_provider(path: str | None = None, *, client: object | None = None):
    """Convenience: load the control file and build its provider, or ``None``."""
    target = load_control_file(path)
    return provider_from_target(target, client=client) if target else None
