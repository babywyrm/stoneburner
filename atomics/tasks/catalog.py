"""Built-in task definitions representing everyday usage patterns."""

from __future__ import annotations

import random

from atomics.models import TaskCategory, TaskDefinition

TASK_CATALOG: list[TaskDefinition] = [
    # ── Web summary ───────────────────────────────────────
    TaskDefinition(
        category=TaskCategory.WEB_SUMMARY,
        name="summarize_tech_article",
        prompt_template=(
            "Summarize the key points of this topic in 3-5 bullet points, "
            "as if briefing a senior engineer: {topic}"
        ),
        weight=2.0,
        max_output_tokens=512,
    ),
    TaskDefinition(
        category=TaskCategory.WEB_SUMMARY,
        name="explain_concept",
        prompt_template=(
            "Explain the following concept clearly and concisely for a "
            "technical audience: {topic}"
        ),
        weight=1.5,
        max_output_tokens=768,
    ),
    # ── Research ──────────────────────────────────────────
    TaskDefinition(
        category=TaskCategory.RESEARCH,
        name="compare_technologies",
        prompt_template=(
            "Compare and contrast {topic} with its main alternatives. "
            "Cover strengths, weaknesses, and ideal use cases."
        ),
        weight=1.5,
        max_output_tokens=1024,
    ),
    TaskDefinition(
        category=TaskCategory.RESEARCH,
        name="deep_dive_topic",
        prompt_template=(
            "Provide a detailed technical analysis of {topic}. "
            "Include architecture, trade-offs, and practical considerations."
        ),
        weight=1.0,
        max_output_tokens=1500,
    ),
    # ── Security news ─────────────────────────────────────
    TaskDefinition(
        category=TaskCategory.SECURITY_NEWS,
        name="cve_analysis",
        prompt_template=(
            "Analyze the security implications of: {topic}. "
            "Cover impact, affected systems, mitigations, and detection strategies."
        ),
        weight=2.0,
        max_output_tokens=1024,
    ),
    TaskDefinition(
        category=TaskCategory.SECURITY_NEWS,
        name="threat_landscape",
        prompt_template=(
            "Describe the current threat landscape around {topic}. "
            "Who are the threat actors, what are their TTPs, and what defenses apply?"
        ),
        weight=1.0,
        max_output_tokens=1024,
    ),
    # ── Pattern extraction ────────────────────────────────
    TaskDefinition(
        category=TaskCategory.PATTERN_EXTRACTION,
        name="extract_patterns",
        prompt_template=(
            "Identify and describe recurring patterns, anti-patterns, or "
            "best practices related to: {topic}. Format as a structured list."
        ),
        weight=1.5,
        max_output_tokens=768,
    ),
    # ── General QA ────────────────────────────────────────
    TaskDefinition(
        category=TaskCategory.GENERAL_QA,
        name="quick_question",
        prompt_template="Answer this concisely: {topic}",
        weight=3.0,
        max_output_tokens=256,
    ),
    TaskDefinition(
        category=TaskCategory.GENERAL_QA,
        name="code_review_snippet",
        prompt_template=(
            "Review the following code concept and suggest improvements: {topic}"
        ),
        weight=1.5,
        max_output_tokens=512,
    ),
]

# Rotating topic pools so prompts vary across runs
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
    ],
    TaskCategory.RESEARCH: [
        "SQLite vs DuckDB for embedded analytics",
        "Kubernetes operators vs Helm charts for application deployment",
        "LLM quantization techniques and accuracy trade-offs",
        "WASM runtimes: Wasmtime vs Wasmer vs WasmEdge",
        "Event sourcing vs traditional CRUD architectures",
        "Vector databases: Pinecone vs Milvus vs pgvector",
    ],
    TaskCategory.SECURITY_NEWS: [
        "XZ Utils backdoor and supply chain trust",
        "HTTP/2 rapid reset DDoS attack vector",
        "OAuth 2.0 token theft via open redirectors",
        "Container escape via kernel vulnerabilities",
        "DNS rebinding attacks against internal services",
        "Prompt injection in LLM-integrated applications",
    ],
    TaskCategory.PATTERN_EXTRACTION: [
        "Retry and circuit-breaker patterns in distributed systems",
        "Secret management patterns in CI/CD pipelines",
        "API versioning strategies",
        "Infrastructure-as-code testing patterns",
        "Observability anti-patterns in microservices",
    ],
    TaskCategory.GENERAL_QA: [
        "What is the CAP theorem and how does it apply to distributed databases?",
        "How does TLS 1.3 differ from TLS 1.2 in its handshake?",
        "What are the OWASP Top 10 for 2025?",
        "Explain the difference between symmetric and asymmetric encryption",
        "What is a race condition and how do you prevent it?",
        "How does memory-safe language adoption reduce CVE counts?",
    ],
}


def get_weighted_task() -> tuple[TaskDefinition, str]:
    """Select a random task weighted by priority and pick a topic for it."""
    weights = [t.weight for t in TASK_CATALOG]
    task = random.choices(TASK_CATALOG, weights=weights, k=1)[0]
    topics = TOPIC_POOLS.get(task.category, ["general software engineering"])
    topic = random.choice(topics)
    return task, topic
