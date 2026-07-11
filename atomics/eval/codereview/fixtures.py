"""Secure-code-review fixtures — can the model find planted vulnerabilities?

Each fixture presents a short piece of code (a full snippet or a unified diff)
and asks the model to perform a security review. Most fixtures contain a single
planted vulnerability with a known CWE; a few are deliberately *clean* so we can
measure false positives (a reviewer that screams "vulnerable!" at safe code is
as useless as one that misses real bugs).

The scorer judges each review against the fixture:
  vulnerable fixture  → detected (model named the planted class) or missed
  clean fixture       → clean-pass (no invented vuln) or false-positive

Rollups: detection_rate (on vulnerable), false_positive_rate (on clean), and an
overall F1-style review score. `mode="diff"` fixtures test review of a change in
isolation — the common real-world PR-review setting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class SecureCodeFixture:
    id: str
    language: str
    mode: Literal["snippet", "diff"]
    code: str
    is_vulnerable: bool
    cwe: str  # e.g. "CWE-89 SQL Injection"; "" for clean fixtures
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL", "NONE"]
    # Concepts/keywords a correct review should surface for a vulnerable fixture.
    expected_findings: list[str] = field(default_factory=list)
    max_output_tokens: int = 768


SECURE_CODE_FIXTURES: list[SecureCodeFixture] = [
    # ── SQL INJECTION (snippet) ───────────────────────────────────────────────
    SecureCodeFixture(
        id="scr-01",
        language="python",
        mode="snippet",
        cwe="CWE-89 SQL Injection",
        severity="CRITICAL",
        is_vulnerable=True,
        code=(
            "def get_user(db, username):\n"
            "    query = \"SELECT * FROM users WHERE name = '\" + username + \"'\"\n"
            "    return db.execute(query).fetchone()\n"
        ),
        expected_findings=[
            "sql injection",
            "string concatenation into the query",
            "use parameterized queries / bound parameters",
        ],
    ),
    # ── COMMAND INJECTION (snippet) ──────────────────────────────────────────
    SecureCodeFixture(
        id="scr-02",
        language="python",
        mode="snippet",
        cwe="CWE-78 OS Command Injection",
        severity="CRITICAL",
        is_vulnerable=True,
        code=(
            "import os\n"
            "def ping(host):\n"
            "    os.system('ping -c 1 ' + host)\n"
        ),
        expected_findings=[
            "command injection",
            "unsanitized input passed to os.system/shell",
            "use subprocess with an argument list / avoid shell=True",
        ],
    ),
    # ── PATH TRAVERSAL (diff) ─────────────────────────────────────────────────
    SecureCodeFixture(
        id="scr-03",
        language="python",
        mode="diff",
        cwe="CWE-22 Path Traversal",
        severity="HIGH",
        is_vulnerable=True,
        code=(
            "--- a/files.py\n"
            "+++ b/files.py\n"
            "@@ -1,3 +1,4 @@\n"
            " def read_doc(name):\n"
            "-    return open(os.path.join(SAFE_DIR, os.path.basename(name))).read()\n"
            "+    return open(os.path.join(SAFE_DIR, name)).read()\n"
        ),
        expected_findings=[
            "path traversal",
            "the change removed basename/normalization so ../ escapes SAFE_DIR",
            "validate the resolved path stays within SAFE_DIR",
        ],
    ),
    # ── HARDCODED SECRET (diff) ──────────────────────────────────────────────
    SecureCodeFixture(
        id="scr-04",
        language="python",
        mode="diff",
        cwe="CWE-798 Hardcoded Credentials",
        severity="HIGH",
        is_vulnerable=True,
        code=(
            "--- a/config.py\n"
            "+++ b/config.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-API_KEY = os.environ['API_KEY']\n"
            "+API_KEY = 'sk_live_EXAMPLE_not_a_real_key_do_not_use'\n"
        ),
        expected_findings=[
            "hardcoded secret / credential in source",
            "the change replaced an env var with a literal key",
            "load from environment / secret manager; rotate the leaked key",
        ],
    ),
    # ── INSECURE DESERIALIZATION (snippet) ───────────────────────────────────
    SecureCodeFixture(
        id="scr-05",
        language="python",
        mode="snippet",
        cwe="CWE-502 Insecure Deserialization",
        severity="HIGH",
        is_vulnerable=True,
        code=(
            "import pickle\n"
            "def load_session(data: bytes):\n"
            "    return pickle.loads(data)\n"
        ),
        expected_findings=[
            "insecure deserialization",
            "pickle.loads on untrusted data enables code execution",
            "use a safe format like json / validate source",
        ],
    ),
    # ── WEAK CRYPTO (snippet) ────────────────────────────────────────────────
    SecureCodeFixture(
        id="scr-06",
        language="python",
        mode="snippet",
        cwe="CWE-327 Weak Hash for Passwords",
        severity="MEDIUM",
        is_vulnerable=True,
        code=(
            "import hashlib\n"
            "def store_password(pw: str):\n"
            "    return hashlib.md5(pw.encode()).hexdigest()\n"
        ),
        expected_findings=[
            "weak / broken hash for passwords (md5)",
            "no salt; fast hash is brute-forceable",
            "use bcrypt/scrypt/argon2",
        ],
    ),
    # ── CLEAN: parameterized query (false-positive check) ────────────────────
    SecureCodeFixture(
        id="scr-clean-01",
        language="python",
        mode="snippet",
        cwe="",
        severity="NONE",
        is_vulnerable=False,
        code=(
            "def get_user(db, username):\n"
            "    return db.execute(\n"
            "        'SELECT * FROM users WHERE name = ?', (username,)\n"
            "    ).fetchone()\n"
        ),
        expected_findings=[],
    ),
    # ── CLEAN: safe subprocess (false-positive check) ────────────────────────
    SecureCodeFixture(
        id="scr-clean-02",
        language="python",
        mode="diff",
        cwe="",
        severity="NONE",
        is_vulnerable=False,
        code=(
            "--- a/net.py\n"
            "+++ b/net.py\n"
            "@@ -1,3 +1,3 @@\n"
            " def ping(host):\n"
            "-    subprocess.run(['ping', '-c', '1', host])\n"
            "+    subprocess.run(['ping', '-c', '2', host], timeout=5)\n"
        ),
        expected_findings=[],
    ),
]
