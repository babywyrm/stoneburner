"""Multilingual evaluation fixtures — test quality across languages.

10 fixtures covering 8 languages: Spanish, French, German, Portuguese,
Japanese, Chinese (Simplified), Korean, and Arabic. Prompts are in the
target language; gold criteria are language-agnostic concepts that a
correct answer must cover regardless of response language.

These complement the English-only eval fixtures by measuring whether
models maintain factual accuracy, completeness, and coherence when
prompted in other languages.
"""

from __future__ import annotations

from atomics.eval.fixtures import EvalFixture
from atomics.models import TaskComplexity

MULTILINGUAL_FIXTURES: list[EvalFixture] = [
    # ── Spanish ───────────────────────────────────────────────────────────────
    EvalFixture(
        id="ml-01",
        complexity=TaskComplexity.LIGHT,
        prompt=(
            "¿Qué es una inyección SQL y cómo se previene? "
            "Responde de forma concisa para un público técnico."
        ),
        gold_criteria=[
            "SQL injection or inserción de código SQL",
            "user input or entrada del usuario",
            "parameterized queries or prepared statements",
        ],
        max_output_tokens=300,
    ),
    # ── French ────────────────────────────────────────────────────────────────
    EvalFixture(
        id="ml-02",
        complexity=TaskComplexity.LIGHT,
        prompt=(
            "Expliquez la différence entre chiffrement symétrique et asymétrique. "
            "Donnez un exemple concret de chaque type."
        ),
        gold_criteria=[
            "symmetric uses same key or clé unique",
            "asymmetric uses key pair or paire de clés",
            "AES or RSA or concrete algorithm example",
        ],
        max_output_tokens=300,
    ),
    # ── German ────────────────────────────────────────────────────────────────
    EvalFixture(
        id="ml-03",
        complexity=TaskComplexity.MODERATE,
        prompt=(
            "Erklären Sie das Prinzip der geringsten Berechtigung (Least Privilege) "
            "und nennen Sie drei konkrete Anwendungsbeispiele in der IT-Sicherheit."
        ),
        gold_criteria=[
            "least privilege or minimale Berechtigung",
            "only necessary permissions",
            "at least two concrete examples",
        ],
        max_output_tokens=400,
    ),
    # ── Portuguese ────────────────────────────────────────────────────────────
    EvalFixture(
        id="ml-04",
        complexity=TaskComplexity.LIGHT,
        prompt=(
            "O que é autenticação multifator (MFA) e por que é importante? "
            "Dê exemplos de fatores utilizados."
        ),
        gold_criteria=[
            "multiple factors or múltiplos fatores",
            "something you know, have, are",
            "password plus token or biometric",
        ],
        max_output_tokens=300,
    ),
    # ── Japanese ──────────────────────────────────────────────────────────────
    EvalFixture(
        id="ml-05",
        complexity=TaskComplexity.MODERATE,
        prompt=(
            "コンテナ技術（Docker）と仮想マシンの違いを説明してください。"
            "それぞれのメリットとデメリットを挙げてください。"
        ),
        gold_criteria=[
            "containers share kernel or カーネル共有",
            "VMs have full OS or 完全なOS",
            "containers are lighter or 軽量",
            "trade-offs or isolation difference",
        ],
        max_output_tokens=500,
    ),
    # ── Chinese (Simplified) ──────────────────────────────────────────────────
    EvalFixture(
        id="ml-06",
        complexity=TaskComplexity.MODERATE,
        prompt=(
            "请解释零信任安全模型的核心原则。"
            "与传统的边界安全模型相比，零信任有哪些优势？"
        ),
        gold_criteria=[
            "never trust always verify or 永不信任始终验证",
            "no implicit trust or 无隐式信任",
            "microsegmentation or identity-based",
            "advantage over perimeter model",
        ],
        max_output_tokens=500,
    ),
    # ── Korean ────────────────────────────────────────────────────────────────
    EvalFixture(
        id="ml-07",
        complexity=TaskComplexity.LIGHT,
        prompt=(
            "API 보안에서 OAuth 2.0의 역할을 설명하세요. "
            "주요 흐름(flow)을 간단히 설명해 주세요."
        ),
        gold_criteria=[
            "OAuth 2.0 authorization framework",
            "access token",
            "authorization code flow or client credentials",
        ],
        max_output_tokens=400,
    ),
    # ── Arabic ────────────────────────────────────────────────────────────────
    EvalFixture(
        id="ml-08",
        complexity=TaskComplexity.LIGHT,
        prompt=(
            "ما هو هجوم التصيد الاحتيالي (Phishing)؟ "
            "اذكر ثلاث طرق للحماية منه."
        ),
        gold_criteria=[
            "phishing or social engineering",
            "fake emails or fraudulent links",
            "at least two protection methods",
        ],
        max_output_tokens=300,
    ),
    # ── Mixed: Spanish technical with English terms ───────────────────────────
    EvalFixture(
        id="ml-09",
        complexity=TaskComplexity.HEAVY,
        prompt=(
            "Diseñe una estrategia de backup 3-2-1 para una empresa con "
            "servidores on-premise y workloads en la nube (AWS). "
            "Incluya RPO, RTO, y herramientas específicas recomendadas."
        ),
        gold_criteria=[
            "3-2-1 rule or tres copias dos medios uno offsite",
            "RPO and RTO definitions",
            "specific tools like AWS S3, Veeam, or rsync",
            "cloud and on-premise coverage",
        ],
        max_output_tokens=600,
    ),
    # ── Mixed: French security analysis ──────────────────────────────────────
    EvalFixture(
        id="ml-10",
        complexity=TaskComplexity.HEAVY,
        prompt=(
            "Analysez les risques de sécurité liés à l'utilisation de modèles "
            "de langage (LLM) en production. Proposez des mesures d'atténuation "
            "pour les trois risques principaux."
        ),
        gold_criteria=[
            "prompt injection",
            "data leakage or sensitive data exposure",
            "hallucination or factual errors",
            "mitigation measures for each risk",
        ],
        max_output_tokens=600,
    ),
]

ALL_MULTILINGUAL_FIXTURES = MULTILINGUAL_FIXTURES
