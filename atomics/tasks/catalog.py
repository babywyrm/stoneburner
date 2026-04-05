"""Built-in task definitions representing everyday usage patterns.

Tasks are tagged with a complexity level:
  - LIGHT: quick, low-token (ez tier and above)
  - MODERATE: standard depth (baseline tier and above)
  - HEAVY: deep research, multi-step reasoning (mega tier only)
"""

from __future__ import annotations

import random

from atomics.models import (
    BurnTier,
    TaskCategory,
    TaskComplexity,
    TaskDefinition,
    TIER_COMPLEXITY_MAP,
)

TASK_CATALOG: list[TaskDefinition] = [
    # ── LIGHT tasks ───────────────────────────────────────
    TaskDefinition(
        category=TaskCategory.GENERAL_QA,
        name="quick_question",
        prompt_template="Answer this concisely: {topic}",
        complexity=TaskComplexity.LIGHT,
        weight=3.0,
        max_output_tokens=256,
    ),
    TaskDefinition(
        category=TaskCategory.WEB_SUMMARY,
        name="summarize_tech_article",
        prompt_template=(
            "Summarize the key points of this topic in 3-5 bullet points, "
            "as if briefing a senior engineer: {topic}"
        ),
        complexity=TaskComplexity.LIGHT,
        weight=2.0,
        max_output_tokens=512,
    ),
    TaskDefinition(
        category=TaskCategory.GENERAL_QA,
        name="definition_lookup",
        prompt_template="Define {topic} in 2-3 sentences for a technical audience.",
        complexity=TaskComplexity.LIGHT,
        weight=2.5,
        max_output_tokens=128,
    ),
    # ── MODERATE tasks ────────────────────────────────────
    TaskDefinition(
        category=TaskCategory.WEB_SUMMARY,
        name="explain_concept",
        prompt_template=(
            "Explain the following concept clearly and concisely for a "
            "technical audience: {topic}"
        ),
        complexity=TaskComplexity.MODERATE,
        weight=1.5,
        max_output_tokens=768,
    ),
    TaskDefinition(
        category=TaskCategory.RESEARCH,
        name="compare_technologies",
        prompt_template=(
            "Compare and contrast {topic} with its main alternatives. "
            "Cover strengths, weaknesses, and ideal use cases."
        ),
        complexity=TaskComplexity.MODERATE,
        weight=1.5,
        max_output_tokens=1024,
    ),
    TaskDefinition(
        category=TaskCategory.SECURITY_NEWS,
        name="cve_analysis",
        prompt_template=(
            "Analyze the security implications of: {topic}. "
            "Cover impact, affected systems, mitigations, and detection strategies."
        ),
        complexity=TaskComplexity.MODERATE,
        weight=2.0,
        max_output_tokens=1024,
    ),
    TaskDefinition(
        category=TaskCategory.PATTERN_EXTRACTION,
        name="extract_patterns",
        prompt_template=(
            "Identify and describe recurring patterns, anti-patterns, or "
            "best practices related to: {topic}. Format as a structured list."
        ),
        complexity=TaskComplexity.MODERATE,
        weight=1.5,
        max_output_tokens=768,
    ),
    TaskDefinition(
        category=TaskCategory.GENERAL_QA,
        name="code_review_snippet",
        prompt_template=(
            "Review the following code concept and suggest improvements: {topic}"
        ),
        complexity=TaskComplexity.MODERATE,
        weight=1.5,
        max_output_tokens=512,
    ),
    # ── HEAVY tasks ───────────────────────────────────────
    TaskDefinition(
        category=TaskCategory.RESEARCH,
        name="deep_dive_topic",
        prompt_template=(
            "Provide a detailed technical analysis of {topic}. "
            "Include architecture, trade-offs, practical considerations, "
            "and a comparison matrix of at least 3 alternatives."
        ),
        complexity=TaskComplexity.HEAVY,
        weight=1.0,
        max_output_tokens=2048,
    ),
    TaskDefinition(
        category=TaskCategory.SECURITY_NEWS,
        name="threat_landscape",
        prompt_template=(
            "Write a comprehensive threat intelligence brief on {topic}. "
            "Cover threat actors, TTPs (MITRE ATT&CK mapped), affected systems, "
            "detection strategies, and recommended mitigations with priority levels."
        ),
        complexity=TaskComplexity.HEAVY,
        weight=1.0,
        max_output_tokens=2048,
    ),
    TaskDefinition(
        category=TaskCategory.RESEARCH,
        name="architecture_review",
        prompt_template=(
            "Design a production architecture for {topic}. Include component diagram "
            "descriptions, data flow, failure modes, scaling strategy, and operational "
            "considerations. Provide concrete technology recommendations."
        ),
        complexity=TaskComplexity.HEAVY,
        weight=0.8,
        max_output_tokens=3000,
    ),
    TaskDefinition(
        category=TaskCategory.PATTERN_EXTRACTION,
        name="security_audit_patterns",
        prompt_template=(
            "Conduct a thorough security audit analysis of {topic}. "
            "Identify attack surfaces, enumerate potential vulnerabilities by category "
            "(injection, auth, crypto, config), suggest remediation for each, "
            "and propose a testing strategy."
        ),
        complexity=TaskComplexity.HEAVY,
        weight=0.8,
        max_output_tokens=3000,
    ),
    TaskDefinition(
        category=TaskCategory.GENERAL_QA,
        name="multi_step_reasoning",
        prompt_template=(
            "Walk through the following problem step-by-step, showing your reasoning "
            "at each stage. Consider edge cases and alternative approaches: {topic}"
        ),
        complexity=TaskComplexity.HEAVY,
        weight=1.0,
        max_output_tokens=2048,
    ),
]

TOPIC_POOLS: dict[TaskCategory, list[str]] = {
    TaskCategory.WEB_SUMMARY: [
        "WebAssembly adoption in production backends",
        "eBPF for observability in cloud-native systems",
        "QUIC protocol vs TCP for modern web applications",
        "Rust in Linux kernel development",
        "Supply chain security for container images",
        "Zero-trust architecture implementation patterns",
        "gRPC vs REST for microservice communication",
        "Database sharding strategies for global scale",
        "Edge computing with Cloudflare Workers vs AWS Lambda@Edge",
        "Service mesh adoption: Istio vs Linkerd vs Cilium",
    ],
    TaskCategory.RESEARCH: [
        "SQLite vs DuckDB for embedded analytics",
        "Kubernetes operators vs Helm charts for application deployment",
        "LLM quantization techniques and accuracy trade-offs",
        "WASM runtimes: Wasmtime vs Wasmer vs WasmEdge",
        "Event sourcing vs traditional CRUD architectures",
        "Vector databases: Pinecone vs Milvus vs pgvector",
        "Building a multi-tenant SaaS platform on Kubernetes",
        "Real-time data pipelines: Kafka vs Pulsar vs Redpanda",
        "Implementing RBAC vs ABAC vs ReBAC authorization models",
        "Comparing container runtimes: containerd vs CRI-O vs gVisor",
    ],
    TaskCategory.SECURITY_NEWS: [
        "XZ Utils backdoor and supply chain trust",
        "HTTP/2 rapid reset DDoS attack vector",
        "OAuth 2.0 token theft via open redirectors",
        "Container escape via kernel vulnerabilities",
        "DNS rebinding attacks against internal services",
        "Prompt injection in LLM-integrated applications",
        "SSRF attacks against cloud metadata services",
        "Kubernetes RBAC misconfigurations and privilege escalation",
        "Typosquatting in PyPI and npm package registries",
        "JWT algorithm confusion attacks",
    ],
    TaskCategory.PATTERN_EXTRACTION: [
        "Retry and circuit-breaker patterns in distributed systems",
        "Secret management patterns in CI/CD pipelines",
        "API versioning strategies",
        "Infrastructure-as-code testing patterns",
        "Observability anti-patterns in microservices",
        "Database migration patterns for zero-downtime deployments",
        "Secure defaults in cloud infrastructure provisioning",
        "Rate limiting and throttling patterns for public APIs",
    ],
    TaskCategory.GENERAL_QA: [
        "What is the CAP theorem and how does it apply to distributed databases?",
        "How does TLS 1.3 differ from TLS 1.2 in its handshake?",
        "What are the OWASP Top 10 for 2025?",
        "Explain the difference between symmetric and asymmetric encryption",
        "What is a race condition and how do you prevent it?",
        "How does memory-safe language adoption reduce CVE counts?",
        "What are the trade-offs between monorepo and polyrepo strategies?",
        "How do you implement mutual TLS between microservices?",
        "Explain the principle of least privilege in cloud IAM design",
        "How does eBPF improve kernel-level security monitoring?",
    ],
}


def get_weighted_task(tier: BurnTier = BurnTier.BASELINE) -> tuple[TaskDefinition, str]:
    """Select a random task weighted by priority, filtered by tier complexity."""
    allowed = TIER_COMPLEXITY_MAP[tier]
    eligible = [t for t in TASK_CATALOG if t.complexity in allowed]
    if not eligible:
        eligible = [t for t in TASK_CATALOG if t.complexity == TaskComplexity.LIGHT]

    weights = [t.weight for t in eligible]
    task = random.choices(eligible, weights=weights, k=1)[0]
    topics = TOPIC_POOLS.get(task.category, ["general software engineering"])
    topic = random.choice(topics)
    return task, topic
