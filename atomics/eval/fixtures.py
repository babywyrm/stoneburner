"""Fixed evaluation fixtures — reproducible, seeded prompts for cross-provider comparison.

These prompts are intentionally stable (not randomised). Running the same set
against every provider gives true apples-to-apples quality scores.

Complexity spread matches the task catalog:
  LIGHT    ~200-400 token responses expected
  MODERATE ~400-800 token responses expected
  HEAVY    ~800-2000 token responses expected

Coverage: security concepts, cloud/infra, LLM/AI, general engineering.

Fixtures ev-16 through ev-25 are reasoning-heavy: multi-step inference, code
tracing, logic problems, and adversarial analysis where chain-of-thought
matters more than factual recall.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from atomics.models import TaskComplexity


@dataclass(frozen=True)
class EvalFixture:
    id: str
    complexity: TaskComplexity
    prompt: str
    gold_criteria: list[str] = field(default_factory=list)
    max_output_tokens: int = 512


EVAL_FIXTURES: list[EvalFixture] = [
    # ── LIGHT ─────────────────────────────────────────────────────────────────
    EvalFixture(
        id="ev-01",
        complexity=TaskComplexity.LIGHT,
        prompt=(
            "What is a supply chain attack? Give a concise 2-3 sentence definition "
            "suitable for a technical audience."
        ),
        gold_criteria=[
            "third-party or dependency compromise",
            "trusted software or vendor",
            "SolarWinds or similar real-world example",
        ],
        max_output_tokens=256,
    ),
    EvalFixture(
        id="ev-02",
        complexity=TaskComplexity.LIGHT,
        prompt=(
            "What does the acronym CVE stand for, what organisation maintains it, "
            "and what is its purpose in security?"
        ),
        gold_criteria=[
            "Common Vulnerabilities and Exposures",
            "MITRE",
            "unique identifier for publicly known vulnerabilities",
        ],
        max_output_tokens=200,
    ),
    EvalFixture(
        id="ev-03",
        complexity=TaskComplexity.LIGHT,
        prompt=(
            "What is the difference between authentication and authorisation? "
            "Give a concrete example of each."
        ),
        gold_criteria=[
            "authentication verifies identity",
            "authorisation determines permissions or access",
            "concrete example such as password login vs role/permission check",
        ],
        max_output_tokens=256,
    ),
    EvalFixture(
        id="ev-04",
        complexity=TaskComplexity.LIGHT,
        prompt=(
            "What is a JSON Web Token (JWT)? Name and describe its three parts."
        ),
        gold_criteria=[
            "header",
            "payload",
            "signature",
            "base64url encoded",
            "stateless authentication",
        ],
        max_output_tokens=300,
    ),
    EvalFixture(
        id="ev-05",
        complexity=TaskComplexity.LIGHT,
        prompt=(
            "Explain the zero-trust security model in 3-4 sentences. "
            "What assumption does it reject?"
        ),
        gold_criteria=[
            "never trust always verify",
            "rejects implicit trust based on network location or perimeter",
            "verify every request regardless of source",
        ],
        max_output_tokens=256,
    ),
    # ── MODERATE ──────────────────────────────────────────────────────────────
    EvalFixture(
        id="ev-06",
        complexity=TaskComplexity.MODERATE,
        prompt=(
            "Explain how Server-Side Request Forgery (SSRF) works. "
            "Describe the attack mechanism and give two distinct real-world impact scenarios."
        ),
        gold_criteria=[
            "attacker causes server to make requests to internal resources",
            "cloud metadata endpoint such as 169.254.169.254",
            "internal service enumeration or credential theft",
            "prevention via allowlist or network segmentation",
        ],
        max_output_tokens=600,
    ),
    EvalFixture(
        id="ev-07",
        complexity=TaskComplexity.MODERATE,
        prompt=(
            "Compare symmetric and asymmetric encryption. Cover: how each works, "
            "performance trade-offs, and when you would use each in practice. "
            "Include at least one algorithm example per type."
        ),
        gold_criteria=[
            "symmetric uses same key for encrypt and decrypt (e.g. AES)",
            "asymmetric uses public/private key pair (e.g. RSA or ECC)",
            "symmetric is faster, asymmetric is used for key exchange",
            "hybrid approach in TLS",
        ],
        max_output_tokens=700,
    ),
    EvalFixture(
        id="ev-08",
        complexity=TaskComplexity.MODERATE,
        prompt=(
            "What is prompt injection in LLM-based systems? "
            "Explain direct vs indirect injection, and describe two concrete defence strategies."
        ),
        gold_criteria=[
            "malicious instructions in user input override system prompt",
            "direct injection via user input",
            "indirect injection via external data sources such as documents or web results",
            "input sanitisation or output validation",
            "privilege separation or sandboxing",
        ],
        max_output_tokens=600,
    ),
    EvalFixture(
        id="ev-09",
        complexity=TaskComplexity.MODERATE,
        prompt=(
            "Explain Kubernetes RBAC. Cover: the four main API objects "
            "(Role, ClusterRole, RoleBinding, ClusterRoleBinding), "
            "how they relate to ServiceAccounts, and give a real-world least-privilege example."
        ),
        gold_criteria=[
            "Role is namespace-scoped, ClusterRole is cluster-wide",
            "RoleBinding links Role to subject within a namespace",
            "ServiceAccount is the identity for pods",
            "least-privilege example such as read-only access to specific resources",
        ],
        max_output_tokens=700,
    ),
    EvalFixture(
        id="ev-10",
        complexity=TaskComplexity.MODERATE,
        prompt=(
            "Compare self-hosted open-weight LLM inference (e.g. Ollama or vLLM) "
            "against hosted API providers (e.g. OpenAI or Anthropic) across: "
            "cost at scale, data privacy, latency, and operational complexity."
        ),
        gold_criteria=[
            "self-hosted has near-zero marginal cost vs per-token API pricing",
            "data never leaves your infrastructure — privacy and compliance advantage",
            "GPU hardware cost or cloud instance cost for self-hosted",
            "operational burden: model updates, scaling, availability",
            "latency: local can be faster at low concurrency, API scales better",
        ],
        max_output_tokens=700,
    ),
    # ── HEAVY ─────────────────────────────────────────────────────────────────
    EvalFixture(
        id="ev-11",
        complexity=TaskComplexity.HEAVY,
        prompt=(
            "Write a threat model for a REST API that handles financial transactions. "
            "Cover: the top 5 attack vectors with severity, recommended controls for each, "
            "and one detection/alerting strategy. Use a structured format."
        ),
        gold_criteria=[
            "authentication and authorisation failures",
            "injection attacks such as SQL injection",
            "man-in-the-middle or TLS stripping",
            "rate limiting and abuse prevention",
            "logging and anomaly detection",
            "structured format with clear sections",
        ],
        max_output_tokens=1500,
    ),
    EvalFixture(
        id="ev-12",
        complexity=TaskComplexity.HEAVY,
        prompt=(
            "Design a secure CI/CD pipeline for deploying containerised applications to Kubernetes. "
            "Address: supply chain security, secrets management, image scanning, "
            "runtime controls, and rollback strategy. Be specific about tooling."
        ),
        gold_criteria=[
            "software bill of materials (SBOM) or image signing (Cosign/Notary)",
            "secrets management with Vault or Sealed Secrets",
            "container image scanning with Trivy, Grype, or Snyk",
            "admission control such as OPA Gatekeeper or Kyverno",
            "rollback with blue/green or canary deployment",
            "specific tool names rather than generic descriptions",
        ],
        max_output_tokens=1800,
    ),
    EvalFixture(
        id="ev-13",
        complexity=TaskComplexity.HEAVY,
        prompt=(
            "Analyse the security and operational trade-offs of running a shared PostgreSQL "
            "database versus per-service databases in a microservices architecture. "
            "Include: attack surface differences, data isolation, migration complexity, "
            "and a concrete recommendation with justification."
        ),
        gold_criteria=[
            "shared DB increases blast radius of SQL injection or credential theft",
            "per-service enforces data isolation at the infrastructure level",
            "least-privilege connection credentials per service",
            "schema migration complexity with independent deployments",
            "clear recommendation with trade-off justification",
        ],
        max_output_tokens=1500,
    ),
    EvalFixture(
        id="ev-14",
        complexity=TaskComplexity.HEAVY,
        prompt=(
            "Explain the MITRE ATT&CK framework: its structure (Tactics, Techniques, "
            "Sub-techniques), how to use it for threat hunting, and walk through a "
            "concrete example mapping an attack scenario (e.g. initial access through "
            "phishing) to specific ATT&CK IDs."
        ),
        gold_criteria=[
            "Tactics are the adversary goal categories",
            "Techniques are how the goal is achieved",
            "Sub-techniques are more specific implementations",
            "concrete ATT&CK IDs such as T1566 for phishing",
            "threat hunting use-case such as detection rule creation",
        ],
        max_output_tokens=1500,
    ),
    EvalFixture(
        id="ev-15",
        complexity=TaskComplexity.HEAVY,
        prompt=(
            "Compare vLLM, Ollama, and llama.cpp for self-hosted LLM inference. "
            "Cover: architecture differences, throughput and latency characteristics, "
            "quantisation support, GPU/CPU requirements, and which workload each is "
            "best suited for. Include a summary comparison table."
        ),
        gold_criteria=[
            "vLLM uses PagedAttention for high throughput batching",
            "Ollama is optimised for ease of use and single-user local inference",
            "llama.cpp targets CPU inference with GGUF quantisation",
            "throughput vs latency trade-offs",
            "GPU VRAM requirements",
            "comparison table or structured summary",
        ],
        max_output_tokens=1800,
    ),
    # ── REASONING-HEAVY ────────────────────────────────────────────────────────
    # These fixtures test multi-step inference, not factual recall.
    # Thinking mode should measurably improve quality on these.
    EvalFixture(
        id="ev-16",
        complexity=TaskComplexity.MODERATE,
        prompt=(
            "A web application has the following middleware chain:\n"
            "1. Rate limiter (100 req/min per IP)\n"
            "2. JWT authentication (validates token, extracts user_id)\n"
            "3. RBAC authorisation (checks user_id against resource ACL)\n"
            "4. Input validation (sanitises request body)\n"
            "5. Business logic handler\n\n"
            "An attacker sends a request with a valid JWT for user A but modifies "
            "the request body to reference user B's resources. "
            "At which layer(s) does this attack succeed or fail? Trace through "
            "each middleware step and explain exactly what happens."
        ),
        gold_criteria=[
            "passes rate limiter (valid IP)",
            "passes JWT auth (valid token for user A)",
            "RBAC check is the critical layer — must verify user_id from JWT matches resource owner",
            "if RBAC only checks 'is authenticated' without ownership, IDOR vulnerability",
            "input validation alone cannot prevent this (body is syntactically valid)",
            "identifies this as an IDOR / broken object-level authorisation issue",
        ],
        max_output_tokens=800,
    ),
    EvalFixture(
        id="ev-17",
        complexity=TaskComplexity.MODERATE,
        prompt=(
            "Read this Python function and find all bugs:\n\n"
            "```python\n"
            "import hashlib\n"
            "import sqlite3\n\n"
            "def authenticate(username, password, db_path):\n"
            "    conn = sqlite3.connect(db_path)\n"
            "    hashed = hashlib.md5(password).hexdigest()\n"
            "    query = f\"SELECT * FROM users WHERE name='{username}' AND pass='{hashed}'\"\n"
            "    result = conn.execute(query).fetchone()\n"
            "    if result:\n"
            "        return True\n"
            "    return False\n"
            "```\n\n"
            "List every security and correctness bug. For each, explain why it's "
            "dangerous and give the fix."
        ),
        gold_criteria=[
            "SQL injection via f-string formatting of username",
            "MD5 is cryptographically broken for password hashing — use bcrypt/argon2",
            "password must be encoded to bytes before hashing (hashlib.md5 requires bytes)",
            "connection is never closed (resource leak)",
            "no salt — identical passwords produce identical hashes",
            "timing attack possible on string comparison (should use hmac.compare_digest)",
        ],
        max_output_tokens=1000,
    ),
    EvalFixture(
        id="ev-18",
        complexity=TaskComplexity.HEAVY,
        prompt=(
            "An attacker has compromised a Kubernetes pod running a web application. "
            "The pod has a mounted ServiceAccount token with the following RBAC:\n\n"
            "- get/list pods in namespace 'prod'\n"
            "- get/list secrets in namespace 'prod'\n"
            "- create pods in namespace 'staging'\n\n"
            "The cluster runs Istio with strict mTLS. CoreDNS is default config. "
            "The node runs containerd with default seccomp profile.\n\n"
            "Map out the complete attack path from this initial foothold. "
            "What can the attacker achieve? What are the pivot opportunities? "
            "What would they target next? Identify at least 3 distinct attack chains."
        ),
        gold_criteria=[
            "read secrets in prod (database creds, API keys, TLS certs)",
            "enumerate pods to discover services and internal architecture",
            "create a privileged pod in staging for container escape",
            "DNS enumeration via CoreDNS to discover cross-namespace services",
            "lateral movement via stolen credentials from secrets",
            "potential node escape via privileged pod if PSP/PSA is weak in staging",
            "mTLS bypass considerations if attacker has the pod's cert",
        ],
        max_output_tokens=2000,
    ),
    EvalFixture(
        id="ev-19",
        complexity=TaskComplexity.MODERATE,
        prompt=(
            "You have three servers: A, B, C. Each makes exactly one statement:\n\n"
            "A: 'Exactly one of us has been compromised.'\n"
            "B: 'A has not been compromised.'\n"
            "C: 'B is lying.'\n\n"
            "Compromised servers always lie. Clean servers always tell the truth.\n\n"
            "Which server(s) are compromised? Show your complete reasoning."
        ),
        gold_criteria=[
            "systematic case analysis (test each combination)",
            "if A is clean, exactly one is compromised",
            "C says B is lying, meaning B is compromised if C is clean",
            "if B is compromised, B lies so A IS compromised — contradicts A being clean",
            "correct answer: A and B are compromised, C is clean",
            "verification: A lies (says 1, but 2 are compromised), B lies (says A clean, but A compromised), C tells truth (B is lying)",
        ],
        max_output_tokens=800,
    ),
    EvalFixture(
        id="ev-20",
        complexity=TaskComplexity.HEAVY,
        prompt=(
            "Trace the execution of this code and give the exact final output:\n\n"
            "```python\n"
            "from collections import defaultdict\n\n"
            "graph = defaultdict(list)\n"
            "edges = [(0,1), (0,2), (1,3), (2,3), (3,4), (2,4)]\n"
            "for u, v in edges:\n"
            "    graph[u].append(v)\n"
            "    graph[v].append(u)\n\n"
            "visited = set()\n"
            "order = []\n\n"
            "def dfs(node):\n"
            "    if node in visited:\n"
            "        return\n"
            "    visited.add(node)\n"
            "    order.append(node)\n"
            "    for neighbor in sorted(graph[node]):\n"
            "        dfs(neighbor)\n\n"
            "dfs(0)\n"
            "print(order)\n"
            "print(sum(order) * len(order))\n"
            "```\n\n"
            "Show the DFS traversal step by step, then give both print outputs."
        ),
        gold_criteria=[
            "adjacency list built correctly as undirected graph",
            "DFS starts at 0, visits sorted neighbors",
            "traversal order: [0, 1, 3, 2, 4]",
            "first print: [0, 1, 3, 2, 4]",
            "sum is 10, length is 5, product is 50",
            "second print: 50",
        ],
        max_output_tokens=1000,
    ),
    EvalFixture(
        id="ev-21",
        complexity=TaskComplexity.MODERATE,
        prompt=(
            "A company's network has the following firewall rules (evaluated top-to-bottom, first match wins):\n\n"
            "1. ALLOW 10.0.0.0/8 → 10.0.1.5:443 (HTTPS to internal API)\n"
            "2. DENY  10.0.2.0/24 → 10.0.1.0/24 (block dev subnet from prod)\n"
            "3. ALLOW 10.0.0.0/8 → 10.0.1.0/24:5432 (PostgreSQL access)\n"
            "4. DENY  any → 10.0.1.0/24 (default deny to prod)\n"
            "5. ALLOW any → any (default allow everything else)\n\n"
            "For each scenario, state which rule matches and whether traffic is allowed:\n"
            "a) 10.0.2.50 → 10.0.1.5:443\n"
            "b) 10.0.2.50 → 10.0.1.10:5432\n"
            "c) 10.0.3.10 → 10.0.1.10:5432\n"
            "d) 192.168.1.5 → 10.0.1.5:443\n"
            "e) 10.0.2.50 → 10.0.5.80:80"
        ),
        gold_criteria=[
            "a) Rule 1 matches (10.0.2.50 is in 10.0.0.0/8, dest matches) — ALLOW",
            "b) Rule 2 matches first (10.0.2.0/24 to 10.0.1.0/24) — DENY",
            "c) Rule 3 matches (10.0.3.10 is in 10.0.0.0/8, not in 10.0.2.0/24) — ALLOW",
            "d) Rule 4 matches (192.168.1.5 is not in 10.0.0.0/8, dest is prod) — DENY",
            "e) Rule 5 matches (dest not in prod subnet) — ALLOW",
        ],
        max_output_tokens=800,
    ),
    EvalFixture(
        id="ev-22",
        complexity=TaskComplexity.HEAVY,
        prompt=(
            "You are reviewing a JWT implementation. The server code does this:\n\n"
            "1. On login: creates JWT with {sub: user_id, role: user.role, exp: now+1h}, "
            "signs with HS256 using a 128-bit key from env var JWT_SECRET\n"
            "2. On API request: verifies signature, checks exp, reads role from token claims\n"
            "3. Admin endpoints: checks if token claim role == 'admin'\n\n"
            "The token is stored in localStorage and sent as Authorization: Bearer header.\n\n"
            "Identify every security weakness in this design. For each, explain the "
            "attack scenario and the specific fix. Rank them by severity."
        ),
        gold_criteria=[
            "algorithm confusion attack (none/HS256 vs RS256)",
            "role stored in token allows privilege escalation if secret is weak",
            "128-bit key is too short for HS256 brute force resistance",
            "localStorage is vulnerable to XSS — use httpOnly cookie",
            "no token revocation mechanism (logout doesn't invalidate)",
            "missing audience (aud) and issuer (iss) claims",
            "no refresh token rotation — 1h session with no way to extend securely",
        ],
        max_output_tokens=1500,
    ),
    EvalFixture(
        id="ev-23",
        complexity=TaskComplexity.MODERATE,
        prompt=(
            "A container image vulnerability scan returns these findings:\n\n"
            "- CVE-2024-3094 (xz-utils backdoor) — CRITICAL, CVSS 10.0\n"
            "- CVE-2023-44487 (HTTP/2 rapid reset) — HIGH, CVSS 7.5\n"
            "- CVE-2023-38545 (curl SOCKS5 heap overflow) — HIGH, CVSS 9.8\n"
            "- CVE-2022-29155 (OpenLDAP SQL injection) — MEDIUM, CVSS 6.5\n"
            "- 47 LOW severity findings in libexpat, zlib, busybox\n\n"
            "The image is a Python 3.11 web API that: uses requests library, "
            "talks to PostgreSQL (not LDAP), runs behind nginx (HTTP/1.1 only), "
            "and does not use xz compression at runtime.\n\n"
            "Triage each finding: is it exploitable in this context? "
            "What's your remediation priority order and why?"
        ),
        gold_criteria=[
            "xz-utils: CRITICAL but likely not exploitable if sshd not exposed — still patch immediately due to severity",
            "curl SOCKS5: depends on whether requests/urllib3 uses system curl — check linkage",
            "HTTP/2 rapid reset: NOT exploitable (nginx is HTTP/1.1 only)",
            "OpenLDAP: NOT exploitable (app uses PostgreSQL not LDAP)",
            "prioritise by actual exploitability not just CVSS score",
            "base image upgrade as first remediation step",
        ],
        max_output_tokens=1000,
    ),
    EvalFixture(
        id="ev-24",
        complexity=TaskComplexity.HEAVY,
        prompt=(
            "Design a token bucket rate limiter that must handle:\n"
            "- 1000 requests/second per API key\n"
            "- Burst allowance of 50 requests\n"
            "- Distributed across 4 server instances\n"
            "- Must survive instance restarts without resetting limits\n"
            "- Sub-millisecond decision latency\n\n"
            "Walk through your design: data structure, storage backend, "
            "the exact algorithm (show pseudocode), how you handle the distributed case, "
            "and what failure modes exist."
        ),
        gold_criteria=[
            "token bucket algorithm: tokens replenish at rate r, bucket capacity b",
            "Redis or similar shared store for distributed state",
            "atomic operations (Lua script or MULTI/EXEC) to prevent race conditions",
            "pseudocode showing consume() with atomic check-and-decrement",
            "handles clock skew or instance failure gracefully",
            "discusses trade-off between consistency and latency",
            "sliding window or leaky bucket as alternative considered",
        ],
        max_output_tokens=1500,
    ),
    EvalFixture(
        id="ev-25",
        complexity=TaskComplexity.MODERATE,
        prompt=(
            "An incident responder finds this in Apache access logs:\n\n"
            "```\n"
            "10.0.2.99 - - [15/May/2026:03:14:22] \"GET /api/users/1 HTTP/1.1\" 200 342\n"
            "10.0.2.99 - - [15/May/2026:03:14:23] \"GET /api/users/2 HTTP/1.1\" 200 342\n"
            "10.0.2.99 - - [15/May/2026:03:14:23] \"GET /api/users/3 HTTP/1.1\" 200 342\n"
            "...\n"
            "10.0.2.99 - - [15/May/2026:03:17:45] \"GET /api/users/8547 HTTP/1.1\" 200 342\n"
            "10.0.2.99 - - [15/May/2026:03:17:46] \"GET /api/users/8548 HTTP/1.1\" 403 28\n"
            "10.0.2.99 - - [15/May/2026:03:18:01] \"POST /api/export HTTP/1.1\" 200 4521887\n"
            "10.0.2.99 - - [15/May/2026:03:18:15] \"DELETE /api/audit-log HTTP/1.1\" 204 0\n"
            "```\n\n"
            "Analyse this incident: What attack is this? What's the timeline? "
            "What data was likely exfiltrated? What's the significance of the final two requests? "
            "What immediate containment and forensic steps would you take?"
        ),
        gold_criteria=[
            "IDOR / broken object-level authorisation (sequential user ID enumeration)",
            "~8547 user records scraped over ~3.5 minutes",
            "403 at 8548 suggests either end of data or privilege boundary",
            "POST /api/export — bulk data exfiltration (4.5MB response)",
            "DELETE /api/audit-log — anti-forensics / evidence destruction",
            "containment: block IP, revoke session, disable export endpoint",
            "forensic: recover audit logs from backups, check if DELETE actually succeeded",
        ],
        max_output_tokens=1000,
    ),
]
