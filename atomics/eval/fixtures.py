"""Fixed evaluation fixtures — reproducible, seeded prompts for cross-provider comparison.

These prompts are intentionally stable (not randomised). Running the same set
against every provider gives true apples-to-apples quality scores.

Complexity spread matches the task catalog:
  LIGHT    ~200-400 token responses expected
  MODERATE ~400-800 token responses expected
  HEAVY    ~800-2000 token responses expected

Coverage: security concepts, cloud/infra, LLM/AI, general engineering.
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
]
