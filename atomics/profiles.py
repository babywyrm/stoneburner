"""Custom target profiles for application-level AI gate testing.

Supports two modes:
- ``ollama``: Hit Ollama /api/generate with a custom system prompt, temperature,
  and num_predict.  Simulates what an app sends to the inference backend.
- ``http``: Hit any arbitrary HTTP endpoint (Flask, Spring, MCP JSON-RPC, …)
  with full control over method, headers, body template, and response parsing.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx


class ProfileError(ValueError):
    """Raised when a profile YAML is missing or invalid."""


@dataclass
class TargetProfile:
    name: str
    type: str  # "ollama" or "http"
    model: str = ""
    prompts: list[str] = field(default_factory=list)
    prompts_file: str | None = None
    classify: dict[str, list[str]] | None = None

    # ollama mode
    ollama_host: str = ""
    system_prompt: str = ""
    temperature: float | None = None
    num_predict: int = 2048

    # http mode
    http_url: str = ""
    http_method: str = "POST"
    http_headers: dict[str, str] = field(default_factory=dict)
    http_body_template: str = ""
    http_timeout: int = 30
    response_format: str = "json"
    response_text_field: str = ""
    response_latency_field: str = ""

    def __post_init__(self) -> None:
        if self.type not in ("ollama", "http"):
            raise ProfileError(
                f"Profile '{self.name}': type must be 'ollama' or 'http', got '{self.type}'"
            )
        if self.type == "ollama" and not self.ollama_host:
            raise ProfileError(f"Profile '{self.name}': ollama type requires ollama.host")
        if self.type == "http" and not self.http_url:
            raise ProfileError(f"Profile '{self.name}': http type requires http.url")


def load_profile(path: str) -> TargetProfile:
    """Load and validate a target profile from a YAML file."""
    p = Path(path)
    if not p.exists():
        raise ProfileError(f"Profile not found: {path}")

    try:
        import yaml
    except ImportError as exc:
        raise ProfileError("PyYAML required for profile loading") from exc

    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise ProfileError(f"{path}: expected a YAML mapping at top level")

    name = raw.get("name", p.stem)
    profile_type = raw.get("type", "")

    kwargs: dict = {"name": name, "type": profile_type}
    kwargs["model"] = raw.get("model", "")
    kwargs["prompts"] = raw.get("prompts", [])
    kwargs["prompts_file"] = raw.get("prompts_file")

    classify_raw = raw.get("classify")
    if classify_raw and isinstance(classify_raw, dict):
        kwargs["classify"] = {k: list(v) for k, v in classify_raw.items()}

    if profile_type == "ollama":
        ollama = raw.get("ollama", {})
        kwargs["ollama_host"] = ollama.get("host", "")
        kwargs["model"] = kwargs["model"] or ollama.get("model", "")
        kwargs["system_prompt"] = ollama.get("system_prompt", "")
        kwargs["temperature"] = ollama.get("temperature")
        kwargs["num_predict"] = ollama.get("num_predict", 2048)
    elif profile_type == "http":
        http = raw.get("http", {})
        kwargs["http_url"] = http.get("url", "")
        kwargs["http_method"] = http.get("method", "POST").upper()
        kwargs["http_headers"] = http.get("headers", {})
        kwargs["http_body_template"] = http.get("body_template", "")
        kwargs["http_timeout"] = http.get("timeout", 30)

        resp = raw.get("response", {})
        kwargs["response_format"] = resp.get("format", "json")
        kwargs["response_text_field"] = resp.get("text_field", "")
        kwargs["response_latency_field"] = resp.get("latency_field", "")

    prompts_file = kwargs.get("prompts_file")
    if prompts_file and not kwargs["prompts"]:
        pf = Path(prompts_file)
        if not pf.is_absolute():
            pf = p.parent / pf
        if pf.exists():
            kwargs["prompts"] = [
                line.strip()
                for line in pf.read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]

    return TargetProfile(**kwargs)


_TEMPLATE_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def render_body(
    profile: TargetProfile,
    prompt: str,
    model: str | None = None,
    num_predict: int | None = None,
) -> str:
    """Render an HTTP body template with variable substitution."""
    template = profile.http_body_template
    if not template:
        return json.dumps({"prompt": prompt})

    values = {
        "prompt": prompt,
        "model": model or profile.model,
        "num_predict": str(num_predict or profile.num_predict),
    }

    def _replace(m: re.Match) -> str:
        key = m.group(1)
        return values.get(key, m.group(0))

    return _TEMPLATE_RE.sub(_replace, template)


def classify_response(profile: TargetProfile, text: str) -> str | None:
    """Map response text to a classification label, or None if no classify rules."""
    if not profile.classify or not text:
        return None
    upper = text.upper()
    for label, patterns in profile.classify.items():
        for pat in patterns:
            if pat.upper() in upper:
                return label
    return "unknown"


def _extract_text(profile: TargetProfile, data: object) -> str:
    """Extract the text field from a parsed response."""
    if profile.response_text_field and isinstance(data, dict):
        return str(data.get(profile.response_text_field, ""))
    if isinstance(data, dict):
        for key in ("response", "text", "result", "output", "decision", "message"):
            if key in data:
                return str(data[key])
        return json.dumps(data)
    return str(data)


def _extract_latency(profile: TargetProfile, data: object) -> float | None:
    """Extract server-reported latency in ms from the response, if configured."""
    if not profile.response_latency_field or not isinstance(data, dict):
        return None
    val = data.get(profile.response_latency_field)
    if val is not None:
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
    return None


async def _single_request_profile(
    client: httpx.AsyncClient,
    profile: TargetProfile,
    prompt: str,
) -> tuple[str, float, str | None]:
    """Fire one request against a profile target.

    Returns ``(response_text, latency_ms, classification)``.
    """
    t0 = time.monotonic()

    if profile.type == "ollama":
        body: dict = {
            "model": profile.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": profile.num_predict},
        }
        if profile.system_prompt:
            body["system"] = profile.system_prompt
        if profile.temperature is not None:
            body["options"]["temperature"] = profile.temperature

        resp = await client.post(
            f"{profile.ollama_host.rstrip('/')}/api/generate",
            json=body,
            timeout=300.0,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "")
        server_lat = data.get("total_duration")
        latency_ms = (server_lat / 1e6) if server_lat else (time.monotonic() - t0) * 1000

    elif profile.type == "http":
        rendered = render_body(profile, prompt)
        headers = dict(profile.http_headers)

        if profile.http_method == "GET":
            resp = await client.get(
                profile.http_url,
                headers=headers,
                timeout=float(profile.http_timeout),
            )
        else:
            if "content-type" not in {k.lower() for k in headers}:
                headers["Content-Type"] = "application/json"
            resp = await client.request(
                profile.http_method,
                profile.http_url,
                content=rendered,
                headers=headers,
                timeout=float(profile.http_timeout),
            )
        resp.raise_for_status()
        latency_ms = (time.monotonic() - t0) * 1000

        if profile.response_format == "json":
            data = resp.json()
            text = _extract_text(profile, data)
            server_lat = _extract_latency(profile, data)
            if server_lat is not None:
                latency_ms = server_lat
        else:
            text = resp.text
    else:
        raise ProfileError(f"Unknown profile type: {profile.type}")

    classification = classify_response(profile, text)
    return text, latency_ms, classification
