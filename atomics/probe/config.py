"""Probe configuration — load and validate probes.yaml.

probes.yaml format:
```yaml
targets:
  - name: nginx-access-logs
    artifact_type: access-log
    source: file
    path: /var/log/nginx/access.log

  - name: ollama-api
    artifact_type: inference-api
    source: http
    url: http://ollama-host:11434/api/tags
    headers:
      Authorization: "Bearer mytoken"   # optional
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

VALID_ARTIFACT_TYPES = frozenset({
    "json-security-report",
    "inference-api",
    "access-log",
    "k8s-audit-log",
    "config-file",
    "api-response",
})


class ProbeConfigError(ValueError):
    """Raised when probes.yaml is missing or invalid."""


@dataclass
class ProbeTarget:
    name: str
    artifact_type: str
    source: Literal["file", "http"]
    path: str | None = None
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    description: str = ""


def load_probe_config(config_path: Path) -> list[ProbeTarget]:
    """Load and validate a probes.yaml file, returning a list of ProbeTarget."""
    if not config_path.exists():
        raise ProbeConfigError(f"Config file not found: {config_path}")

    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ProbeConfigError("PyYAML is required for probe config loading. Install it with: pip install pyyaml") from exc

    try:
        raw = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as exc:
        raise ProbeConfigError(f"YAML parse error in {config_path}: {exc}") from exc

    if not isinstance(raw, dict) or "targets" not in raw:
        raise ProbeConfigError(f"{config_path}: expected a 'targets:' key at the top level.")

    targets: list[ProbeTarget] = []
    for i, entry in enumerate(raw["targets"]):
        name = entry.get("name", f"target-{i}")
        artifact_type = entry.get("artifact_type", "")
        if artifact_type not in VALID_ARTIFACT_TYPES:
            raise ProbeConfigError(
                f"Target '{name}': invalid artifact_type '{artifact_type}'. "
                f"Valid types: {sorted(VALID_ARTIFACT_TYPES)}"
            )
        source = entry.get("source", "file")
        targets.append(ProbeTarget(
            name=name,
            artifact_type=artifact_type,
            source=source,
            path=entry.get("path"),
            url=entry.get("url"),
            headers=entry.get("headers") or {},
            description=entry.get("description", ""),
        ))

    return targets
