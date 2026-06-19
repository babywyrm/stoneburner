"""Deterministic evidence-pack builder: a repo → a single budgeted text blob.

Same repo + same TierConfig always yields byte-identical output (sorted file
ordering, deterministic tail-truncation), so every model in a run sees the same
input and re-runs are reproducible.
"""

from __future__ import annotations

import fnmatch
import hashlib
from dataclasses import dataclass
from pathlib import Path

from atomics.archreview.models import TierConfig

_CHARS_PER_TOKEN = 4

# Files never worth packing regardless of repo (binary / lockfile noise).
_DEFAULT_EXCLUDE = (
    "**/.git/**", "**/node_modules/**", "**/dist/**", "**/build/**",
    "**/*.min.js", "**/*.map", "**/*.lock", "**/*.png", "**/*.jpg",
    "**/*.gif", "**/*.ico", "**/*.svg", "**/*.woff*", "**/*.ttf",
)

# Security-relevant filename hints used to rank files when no explicit priority
# is given. Earlier patterns rank higher.
_RELEVANCE_HINTS = (
    "auth", "login", "session", "middleware", "route", "controller",
    "access", "permission", "crypto", "password", "token", "config",
    "security", "user", "admin", "upload", "query", "db", "sql",
)
_MANIFESTS = ("package.json", "requirements.txt", "pom.xml", "go.mod", "Gemfile")


def estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


@dataclass(frozen=True)
class EvidencePack:
    text: str
    content_hash: str
    file_count: int
    truncated: bool


def _matches_any(rel: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(rel, p.lstrip("*/"))
               for p in patterns)


def _relevance_rank(rel: str, priority: tuple[str, ...]) -> tuple[int, str]:
    # Lower sort key = higher priority. Explicit priority first (by index),
    # then manifests, then relevance-hint hits, then everything else. Ties
    # break on path for determinism.
    for i, pat in enumerate(priority):
        if fnmatch.fnmatch(rel, pat) or rel == pat:
            return (i, rel)
    base = rel.rsplit("/", 1)[-1]
    if base in _MANIFESTS:
        return (1000, rel)
    low = rel.lower()
    for j, hint in enumerate(_RELEVANCE_HINTS):
        if hint in low:
            return (2000 + j, rel)
    return (9000, rel)


def build_pack(repo_path: Path, cfg: TierConfig) -> EvidencePack:
    """Build a deterministic evidence pack for a repo under a TierConfig."""
    repo_path = Path(repo_path)
    exclude = _DEFAULT_EXCLUDE + tuple(cfg.exclude)
    include = tuple(cfg.include)

    files: list[str] = []
    for p in sorted(repo_path.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(repo_path).as_posix()
        if _matches_any(rel, exclude):
            continue
        if include and not _matches_any(rel, include):
            continue
        files.append(rel)

    files.sort(key=lambda r: _relevance_rank(r, tuple(cfg.priority)))

    tree = "\n".join(sorted(files))
    header = f"# REPOSITORY EVIDENCE PACK\n## File tree ({len(files)} files)\n{tree}\n\n## Files\n"

    budget_chars = cfg.budget_tokens * _CHARS_PER_TOKEN
    parts: list[str] = [header]
    used = len(header)
    truncated = False
    packed = 0
    for rel in files:
        try:
            body = (repo_path / rel).read_text(errors="replace")
        except OSError:
            continue
        block = f"\n=== {rel} ===\n{body}\n"
        if used + len(block) > budget_chars:
            remaining = max(0, budget_chars - used)
            if remaining > 200:  # worth including a partial file
                parts.append(block[:remaining])
                packed += 1
            parts.append("\n=== [TRUNCATED: token budget reached] ===\n")
            truncated = True
            break
        parts.append(block)
        used += len(block)
        packed += 1

    text = "".join(parts)
    digest = hashlib.sha256(text.encode()).hexdigest()
    return EvidencePack(text=text, content_hash=digest,
                        file_count=packed, truncated=truncated)
