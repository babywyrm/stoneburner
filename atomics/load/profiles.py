"""Custom target profiles for application-level AI gate testing.

Supports two modes:
- ``ollama``: Hit Ollama /api/generate with a custom system prompt, temperature,
  num_predict, and a bounded ``num_ctx``.  Simulates what an app sends to the
  inference backend.
- ``http``: Hit any arbitrary HTTP endpoint (Flask, Spring, MCP JSON-RPC, …)
  with full control over method, headers, body template, and response parsing.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from atomics.providers.base import BaseProvider, ProviderResponse, compute_tps
from atomics.providers.ollama import DEFAULT_NUM_CTX


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
        from atomics.validation import validate_endpoint_url

        endpoint = self.ollama_host if self.type == "ollama" else self.http_url
        label = "ollama.host" if self.type == "ollama" else "http.url"
        try:
            cleaned = validate_endpoint_url(endpoint, label=label)
        except ValueError as exc:
            raise ProfileError(str(exc)) from exc
        if self.type == "ollama":
            self.ollama_host = cleaned
        else:
            self.http_url = cleaned


def _confined_profile_catalog(root_dir: Path | None = None) -> dict[str, Path]:
    base = (root_dir or Path("profiles")).resolve()
    real_base = os.path.realpath(base)
    catalog: dict[str, Path] = {}
    if not base.is_dir():
        return catalog
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames[:] = [d for d in dirnames if not (Path(dirpath) / d).is_symlink()]
        for fname in filenames:
            if not fname.lower().endswith((".yaml", ".yml")):
                continue
            candidate = Path(dirpath) / fname
            if candidate.is_symlink():
                continue
            real_candidate = os.path.realpath(candidate)
            if not real_candidate.startswith(real_base + os.sep):
                continue
            rel = candidate.relative_to(base).as_posix()
            safe_path = Path(real_candidate)
            catalog[rel] = safe_path
            catalog[f"profiles/{rel}"] = safe_path
    return catalog


def load_confined_profile(profile_ref: str, *, root_dir: Path | None = None) -> TargetProfile:
    """Resolve `profile_ref` via a catalog of non-symlink YAML files under `profiles/`."""
    key = profile_ref.strip().replace("\\", "/").lstrip("./")
    if not key or ".." in key.split("/") or not key.lower().endswith((".yaml", ".yml")):
        raise ProfileError("profile must be a relative .yaml/.yml path under profiles/")
    matched = _confined_profile_catalog(root_dir).get(key)
    if matched is None:
        raise ProfileError(f"Profile not found under profiles/: {profile_ref}")
    return load_profile(matched)


def load_profile(path: str | Path) -> TargetProfile:
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

    name = str(raw.get("name", p.stem))
    profile_type = str(raw.get("type", ""))
    model = str(raw.get("model", ""))
    raw_prompts = raw.get("prompts")
    prompts = [str(item) for item in raw_prompts] if isinstance(raw_prompts, list) else []
    prompts_file = raw.get("prompts_file")

    classify: dict[str, list[str]] | None = None
    classify_raw = raw.get("classify")
    if classify_raw and isinstance(classify_raw, dict):
        classify = {
            str(k): [str(item) for item in v] if isinstance(v, list) else []
            for k, v in classify_raw.items()
        }

    ollama_host = ""
    system_prompt = ""
    temperature: float | None = None
    num_predict = 2048
    http_url = ""
    http_method = "POST"
    http_headers: dict[str, str] = {}
    http_body_template = ""
    http_timeout = 30
    response_format = "json"
    response_text_field = ""
    response_latency_field = ""

    if profile_type == "ollama":
        ollama = raw.get("ollama")
        if isinstance(ollama, dict):
            ollama_host = str(ollama.get("host", ""))
            model = model or str(ollama.get("model", ""))
            system_prompt = str(ollama.get("system_prompt", ""))
            raw_temp = ollama.get("temperature")
            temperature = float(raw_temp) if isinstance(raw_temp, (int, float)) else None
            raw_np = ollama.get("num_predict", 2048)
            num_predict = int(raw_np) if isinstance(raw_np, (int, float)) else 2048
    elif profile_type == "http":
        http = raw.get("http")
        if isinstance(http, dict):
            http_url = str(http.get("url", ""))
            http_method = str(http.get("method", "POST")).upper()
            raw_headers = http.get("headers")
            if isinstance(raw_headers, dict):
                http_headers = {str(k): str(v) for k, v in raw_headers.items()}
            http_body_template = str(http.get("body_template", ""))
            raw_timeout = http.get("timeout", 30)
            http_timeout = int(raw_timeout) if isinstance(raw_timeout, (int, float)) else 30

        resp = raw.get("response")
        if isinstance(resp, dict):
            response_format = str(resp.get("format", "json"))
            response_text_field = str(resp.get("text_field", ""))
            response_latency_field = str(resp.get("latency_field", ""))

    if prompts_file is not None and not isinstance(prompts_file, str):
        raise ProfileError(f"{p}: prompts_file must be a string path")
    if prompts_file and not prompts:
        raw_pf = p.parent / prompts_file
        real_parent = os.path.realpath(p.parent)
        real_pf = os.path.realpath(raw_pf)
        if raw_pf.is_symlink() or not real_pf.startswith(real_parent + os.sep):
            raise ProfileError(f"{p}: prompts_file must stay inside the profile's directory")
        pf = Path(real_pf)
        if pf.is_file():
            prompts = [
                line.strip()
                for line in pf.read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]

    return TargetProfile(
        name=name,
        type=profile_type,
        model=model,
        prompts=prompts,
        prompts_file=prompts_file,
        classify=classify,
        ollama_host=ollama_host,
        system_prompt=system_prompt,
        temperature=temperature,
        num_predict=num_predict,
        http_url=http_url,
        http_method=http_method,
        http_headers=http_headers,
        http_body_template=http_body_template,
        http_timeout=http_timeout,
        response_format=response_format,
        response_text_field=response_text_field,
        response_latency_field=response_latency_field,
    )


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

    # ponytail: "looks like JSON" is the leading brace; a JSON template that
    # starts otherwise gets raw values. Upgrade: a response.format-style flag.
    as_json = template.lstrip().startswith(("{", "["))

    def _replace(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in values:
            return m.group(0)
        return str(json.dumps(values[key])[1:-1]) if as_json else values[key]

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
        from atomics.validation import validate_endpoint_url

        host = validate_endpoint_url(profile.ollama_host, label="ollama.host")
        options: dict[str, int | float] = {
            "num_predict": profile.num_predict,
            "num_ctx": DEFAULT_NUM_CTX,
        }
        if profile.temperature is not None:
            options["temperature"] = profile.temperature
        body: dict[str, object] = {
            "model": profile.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if profile.system_prompt:
            body["system"] = profile.system_prompt

        resp = await client.post(
            f"{host}/api/generate",
            json=body,
            timeout=300.0,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "")
        server_lat = data.get("total_duration")
        latency_ms = (server_lat / 1e6) if server_lat else (time.monotonic() - t0) * 1000

    elif profile.type == "http":
        from atomics.validation import validate_endpoint_url

        url = validate_endpoint_url(profile.http_url, label="http.url")
        rendered = render_body(profile, prompt)
        headers = dict(profile.http_headers)

        if profile.http_method == "GET":
            resp = await client.get(
                url,
                headers=headers,
                timeout=float(profile.http_timeout),
            )
        else:
            if "content-type" not in {k.lower() for k in headers}:
                headers["Content-Type"] = "application/json"
            resp = await client.request(
                profile.http_method,
                url,
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


class ProfileProvider(BaseProvider):
    """Wrap a `TargetProfile` as a `BaseProvider` for prose security suites.

    Lets `adversarial`, `refusal`, `redblue`, and `battery` evaluate an
    application-level HTTP or Ollama gate with the same fixtures and judge
    pipeline used against raw providers. Tool schemas are not supported
    (`supports_tools = False`).
    """

    supports_tools = False

    def __init__(
        self,
        profile: TargetProfile,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._profile = profile
        self._client = client or httpx.AsyncClient()

    @property
    def profile(self) -> TargetProfile:
        return self._profile

    @property
    def name(self) -> str:
        return f"profile:{self._profile.name}"

    @property
    def default_model(self) -> str:
        return self._profile.model or self._profile.name

    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
        thinking: bool | None = None,
        thinking_budget: int | None = None,
        temperature: float | None = None,
        effort: str | None = None,
        reasoning_mode: str | None = None,
    ) -> ProviderResponse:
        del thinking, thinking_budget, effort, reasoning_mode
        t0 = time.monotonic()
        effective_model = model or self._profile.model or self._profile.name
        input_tokens = max(1, len(prompt) // 4)
        output_tokens = 0

        if self._profile.type == "ollama":
            from atomics.validation import validate_endpoint_url

            host = validate_endpoint_url(self._profile.ollama_host, label="ollama.host")
            num_predict = max_tokens if max_tokens > 0 else self._profile.num_predict
            options: dict[str, int | float] = {
                "num_predict": num_predict,
                "num_ctx": DEFAULT_NUM_CTX,
            }
            eff_temp = temperature if temperature is not None else self._profile.temperature
            if eff_temp is not None:
                options["temperature"] = eff_temp
            body: dict[str, object] = {
                "model": effective_model,
                "prompt": prompt,
                "stream": False,
                "options": options,
            }
            eff_system = self._profile.system_prompt or system
            if eff_system:
                body["system"] = eff_system
            resp = await self._client.post(f"{host}/api/generate", json=body, timeout=300.0)
            resp.raise_for_status()
            data = resp.json()
            text = str(data.get("response", ""))
            server_lat = data.get("total_duration")
            latency_ms = float(server_lat) / 1e6 if server_lat else (time.monotonic() - t0) * 1000
            prompt_eval = data.get("prompt_eval_count")
            eval_cnt = data.get("eval_count")
            input_tokens = int(prompt_eval) if isinstance(prompt_eval, int) else input_tokens
            output_tokens = (
                int(eval_cnt)
                if isinstance(eval_cnt, int)
                else (max(1, len(text) // 4) if text else 0)
            )
        elif self._profile.type == "http":
            from atomics.validation import validate_endpoint_url

            url = validate_endpoint_url(self._profile.http_url, label="http.url")
            eff_prompt = (
                f"{system}\n\n{prompt}" if system and not self._profile.system_prompt else prompt
            )
            rendered = render_body(
                self._profile,
                eff_prompt,
                model=model,
                num_predict=max_tokens,
            )
            headers = dict(self._profile.http_headers)
            if self._profile.http_method == "GET":
                resp = await self._client.get(
                    url,
                    headers=headers,
                    timeout=float(self._profile.http_timeout),
                )
            else:
                if "content-type" not in {k.lower() for k in headers}:
                    headers["Content-Type"] = "application/json"
                resp = await self._client.request(
                    self._profile.http_method,
                    url,
                    content=rendered,
                    headers=headers,
                    timeout=float(self._profile.http_timeout),
                )
            resp.raise_for_status()
            latency_ms = (time.monotonic() - t0) * 1000
            if self._profile.response_format == "json":
                data = resp.json()
                text = _extract_text(self._profile, data)
                server_lat = _extract_latency(self._profile, data)
                if server_lat is not None:
                    latency_ms = server_lat
            else:
                text = resp.text
            output_tokens = max(1, len(text) // 4) if text else 0
        else:
            raise ProfileError(f"Unknown profile type: {self._profile.type}")

        classification = classify_response(self._profile, text)
        return ProviderResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            model=effective_model,
            latency_ms=latency_ms,
            estimated_cost_usd=0.0,
            tokens_per_second=compute_tps(output_tokens, latency_ms),
            raw={"classification": classification} if classification is not None else None,
        )

    async def health_check(self) -> bool:
        try:
            if self._profile.type == "ollama":
                from atomics.validation import validate_endpoint_url

                host = validate_endpoint_url(self._profile.ollama_host, label="ollama.host")
                resp = await self._client.get(f"{host}/api/tags", timeout=5.0)
                return resp.status_code == 200
            await _single_request_profile(self._client, self._profile, "healthcheck")
            return True
        except (httpx.HTTPError, ProfileError, ValueError):
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
