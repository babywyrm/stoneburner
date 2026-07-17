"""RAG evaluation fixtures — 20 fixed prompts with context chunks.

10 security-focused, 10 general technical. Each fixture has context chunks
labeled as relevant or distractor, and gold criteria for grounding assessment.
"""

from __future__ import annotations

from atomics.eval.rag import RAGChunk, RAGFixture
from atomics.models import TaskComplexity

RAG_FIXTURES: list[RAGFixture] = [
    # ── SECURITY FIXTURES ─────────────────────────────────────────────────────
    RAGFixture(
        id="rag-01",
        complexity=TaskComplexity.LIGHT,
        question="What is the severity and impact of CVE-2026-3891?",
        context_chunks=[
            RAGChunk(
                content=(
                    "CVE-2026-3891: A critical remote code execution vulnerability in "
                    "libxml2 versions prior to 2.12.5. An attacker can craft a malicious "
                    "XML document that triggers a heap buffer overflow during parsing. "
                    "CVSS 3.1 Base Score: 9.8. Affected: libxml2 < 2.12.5. "
                    "Fix: upgrade to libxml2 >= 2.12.5."
                ),
                label="relevant",
                source="CVE-2026-3891.md",
            ),
            RAGChunk(
                content=(
                    "CVE-2025-0001: A medium-severity XSS vulnerability in the admin "
                    "dashboard of FooBar CMS v3.2. Requires authenticated access. "
                    "CVSS 3.1 Base Score: 5.4."
                ),
                label="distractor",
                source="CVE-2025-0001.md",
            ),
        ],
        gold_criteria=[
            "critical severity",
            "remote code execution",
            "libxml2",
            "heap buffer overflow",
            "CVSS 9.8",
        ],
        max_output_tokens=256,
    ),
    RAGFixture(
        id="rag-02",
        complexity=TaskComplexity.MODERATE,
        question=(
            "Based on the incident report, what was the initial access vector "
            "and what lateral movement techniques were used?"
        ),
        context_chunks=[
            RAGChunk(
                content=(
                    "Incident Report IR-2026-047: Initial access was achieved via "
                    "a spearphishing email with a malicious macro document sent to "
                    "the finance department on 2026-03-15. The attacker established "
                    "persistence via a scheduled task and moved laterally using "
                    "PsExec and stolen Kerberos tickets (Pass-the-Ticket). Domain "
                    "admin credentials were obtained from a memory dump of LSASS."
                ),
                label="relevant",
                source="IR-2026-047.md",
            ),
            RAGChunk(
                content=(
                    "Incident Report IR-2026-012: A DDoS attack against the public "
                    "API gateway caused 4 hours of downtime. Mitigated with WAF "
                    "rate limiting rules."
                ),
                label="distractor",
                source="IR-2026-012.md",
            ),
            RAGChunk(
                content=(
                    "Post-incident: the LSASS dump was performed using Mimikatz "
                    "variant loaded via reflective DLL injection. Credential "
                    "harvesting took place across 3 domain controllers."
                ),
                label="relevant",
                source="IR-2026-047-addendum.md",
            ),
        ],
        gold_criteria=[
            "spearphishing",
            "malicious macro",
            "PsExec",
            "Pass-the-Ticket or Kerberos",
            "LSASS or credential dump",
        ],
        max_output_tokens=512,
    ),
    RAGFixture(
        id="rag-03",
        complexity=TaskComplexity.LIGHT,
        question="Does the security policy allow the use of personal devices for accessing production systems?",
        context_chunks=[
            RAGChunk(
                content=(
                    "Section 4.2 - Remote Access Policy: All access to production "
                    "environments must originate from company-managed devices enrolled "
                    "in the MDM solution. Personal devices (BYOD) are permitted for "
                    "email and collaboration tools only. VPN access requires a valid "
                    "client certificate issued by the internal CA."
                ),
                label="relevant",
                source="security-policy-v3.md",
            ),
            RAGChunk(
                content=(
                    "Section 7.1 - Password Policy: All user accounts must use "
                    "passwords with a minimum of 14 characters. MFA is required "
                    "for all externally-facing services."
                ),
                label="distractor",
                source="security-policy-v3.md",
            ),
        ],
        gold_criteria=[
            "personal devices not allowed for production",
            "company-managed devices required",
            "BYOD only for email and collaboration",
        ],
        max_output_tokens=256,
    ),
    RAGFixture(
        id="rag-04",
        complexity=TaskComplexity.MODERATE,
        question="What indicators of compromise should we look for based on this threat intelligence?",
        context_chunks=[
            RAGChunk(
                content=(
                    "Threat Intelligence Report TI-2026-089: APT group 'SilverFox' "
                    "targets financial sector. Known C2 domains: evil-update.com, "
                    "cdn-analytics.net. Malware hashes (SHA-256): "
                    "a1b2c3d4e5f6...78 (loader), f9e8d7c6b5a4...21 (RAT). "
                    "Behavior: DLL sideloading via legitimate signed binaries, "
                    "DNS tunneling for exfiltration, registry run key persistence."
                ),
                label="relevant",
                source="TI-2026-089.md",
            ),
            RAGChunk(
                content=(
                    "Weekly vulnerability scan results: 47 medium findings, "
                    "3 high findings. Top issue: outdated OpenSSL on 12 hosts."
                ),
                label="distractor",
                source="vuln-scan-weekly.md",
            ),
        ],
        gold_criteria=[
            "C2 domains",
            "malware hashes",
            "DLL sideloading",
            "DNS tunneling",
            "registry persistence",
        ],
        max_output_tokens=512,
    ),
    RAGFixture(
        id="rag-05",
        complexity=TaskComplexity.LIGHT,
        question="What is the remediation status for CVE-2026-5555?",
        context_chunks=[
            RAGChunk(
                content=(
                    "CVE-2026-3891 remediation: Patched on 2026-04-01 across all "
                    "production hosts. Verified by vulnerability rescan."
                ),
                label="distractor",
                source="patch-status.md",
            ),
            RAGChunk(
                content=(
                    "Quarterly patch report Q2 2026: 142 vulnerabilities remediated. "
                    "Outstanding: CVE-2026-7890 (awaiting vendor patch)."
                ),
                label="distractor",
                source="quarterly-patch-q2.md",
            ),
        ],
        gold_criteria=[
            "not available",
            "not in the provided context",
        ],
        context_contains_answer=False,
        max_output_tokens=200,
    ),
    RAGFixture(
        id="rag-06",
        complexity=TaskComplexity.MODERATE,
        question="What credentials were exposed in the audit log and what systems were affected?",
        context_chunks=[
            RAGChunk(
                content=(
                    "Audit log analysis 2026-05-20: Detected plaintext AWS access "
                    "key AKIA... in application logs on web-server-03 and "
                    "web-server-07. The key belongs to service account 'deploy-bot' "
                    "with S3 and EC2 permissions. Key was last rotated 180 days ago."
                ),
                label="relevant",
                source="audit-findings-2026-05.md",
            ),
            RAGChunk(
                content=(
                    "Log rotation policy: Application logs are rotated daily and "
                    "retained for 90 days. Logs older than 90 days are archived "
                    "to cold storage."
                ),
                label="distractor",
                source="log-policy.md",
            ),
        ],
        gold_criteria=[
            "AWS access key",
            "plaintext in application logs",
            "web-server-03 and web-server-07",
            "deploy-bot service account",
            "S3 and EC2 permissions",
        ],
        max_output_tokens=400,
    ),
    RAGFixture(
        id="rag-07",
        complexity=TaskComplexity.HEAVY,
        question="Analyze the SBOM and identify which dependencies have known critical vulnerabilities.",
        context_chunks=[
            RAGChunk(
                content=(
                    "SBOM excerpt (CycloneDX): \n"
                    "- log4j-core 2.14.1 (CVE-2021-44228, CVSS 10.0, critical)\n"
                    "- jackson-databind 2.13.0 (CVE-2022-42003, CVSS 7.5, high)\n"
                    "- spring-boot 2.7.18 (no known critical CVEs)\n"
                    "- commons-text 1.9 (CVE-2022-42889, CVSS 9.8, critical)"
                ),
                label="relevant",
                source="sbom-backend.json",
            ),
            RAGChunk(
                content=(
                    "Frontend dependencies: react 18.2.0, webpack 5.88.0, "
                    "typescript 5.1.6. No known vulnerabilities."
                ),
                label="distractor",
                source="sbom-frontend.json",
            ),
        ],
        gold_criteria=[
            "log4j-core 2.14.1",
            "CVE-2021-44228",
            "commons-text 1.9",
            "CVE-2022-42889",
            "critical severity",
        ],
        max_output_tokens=600,
    ),
    RAGFixture(
        id="rag-08",
        complexity=TaskComplexity.LIGHT,
        question="What are the steps to rotate the database credentials according to the runbook?",
        context_chunks=[
            RAGChunk(
                content=(
                    "Runbook: Database Credential Rotation\n"
                    "1. Generate new credentials in Vault: vault write database/rotate-root/prod-db\n"
                    "2. Update the application config via Consul KV\n"
                    "3. Rolling restart of app pods: kubectl rollout restart deployment/api\n"
                    "4. Verify connectivity: run healthcheck endpoint\n"
                    "5. Revoke old credentials in Vault after 1 hour grace period"
                ),
                label="relevant",
                source="runbook-db-rotation.md",
            ),
        ],
        gold_criteria=[
            "Vault",
            "rotate-root",
            "Consul KV",
            "rolling restart",
            "revoke old credentials",
        ],
        max_output_tokens=400,
    ),
    RAGFixture(
        id="rag-09",
        complexity=TaskComplexity.MODERATE,
        question="Based on these alerts, is this a true positive or false positive, and why?",
        context_chunks=[
            RAGChunk(
                content=(
                    "SIEM Alert: Possible data exfiltration detected. "
                    "Source: workstation-42 (user: j.smith, dept: engineering). "
                    "Destination: upload.dropbox.com. Volume: 2.3 GB over 45 minutes. "
                    "Time: 2026-06-15 02:30 UTC (outside business hours)."
                ),
                label="relevant",
                source="siem-alert-9981.json",
            ),
            RAGChunk(
                content=(
                    "HR record: j.smith submitted resignation effective 2026-06-30. "
                    "Currently in 2-week notice period. Has access to source code "
                    "repositories and design documents."
                ),
                label="relevant",
                source="hr-notice.md",
            ),
            RAGChunk(
                content=(
                    "Network baseline: engineering department averages 500MB daily "
                    "external upload. Top destinations: github.com, npm registry."
                ),
                label="relevant",
                source="network-baseline.md",
            ),
        ],
        gold_criteria=[
            "likely true positive",
            "departing employee",
            "outside business hours",
            "volume exceeds baseline",
            "data exfiltration risk",
        ],
        max_output_tokens=512,
    ),
    RAGFixture(
        id="rag-10",
        complexity=TaskComplexity.HEAVY,
        question="Synthesize a patch guidance memo from these advisories for the security team.",
        context_chunks=[
            RAGChunk(
                content=(
                    "Advisory 1: Critical RCE in OpenSSH < 9.8 (CVE-2026-8800). "
                    "Affects all Linux hosts. Exploit is public. Patch immediately. "
                    "Workaround: disable SSH agent forwarding."
                ),
                label="relevant",
                source="advisory-openssh.md",
            ),
            RAGChunk(
                content=(
                    "Advisory 2: High-severity privilege escalation in sudo < 1.9.16 "
                    "(CVE-2026-8801). Local attacker can gain root. Patch available. "
                    "No known workaround."
                ),
                label="relevant",
                source="advisory-sudo.md",
            ),
            RAGChunk(
                content=(
                    "Advisory 3: Medium information disclosure in nginx < 1.27 "
                    "(CVE-2026-8802). Leaks server version in error pages. "
                    "Low impact. Patch at convenience."
                ),
                label="relevant",
                source="advisory-nginx.md",
            ),
            RAGChunk(
                content=(
                    "Last month's completed patches: PostgreSQL 16.2, Node.js 20.12, "
                    "Docker 25.0.4. All verified."
                ),
                label="distractor",
                source="patch-history.md",
            ),
        ],
        gold_criteria=[
            "OpenSSH CVE-2026-8800 critical",
            "sudo CVE-2026-8801 high",
            "nginx CVE-2026-8802 medium",
            "prioritize by severity",
            "workaround for SSH agent forwarding",
        ],
        max_output_tokens=800,
    ),
    # ── GENERAL TECHNICAL FIXTURES ────────────────────────────────────────────
    RAGFixture(
        id="rag-11",
        complexity=TaskComplexity.LIGHT,
        question="How do I authenticate to the user management API?",
        context_chunks=[
            RAGChunk(
                content=(
                    "API Reference: User Management\n"
                    "Authentication: Bearer token in the Authorization header. "
                    "Obtain a token via POST /auth/token with client_id and "
                    "client_secret in the request body. Tokens expire after 1 hour. "
                    "Rate limit: 100 requests per minute per token."
                ),
                label="relevant",
                source="api-docs/auth.md",
            ),
            RAGChunk(
                content=(
                    "API Reference: Billing\n"
                    "Endpoint: GET /billing/invoices. Requires billing:read scope. "
                    "Returns paginated list of invoices."
                ),
                label="distractor",
                source="api-docs/billing.md",
            ),
        ],
        gold_criteria=[
            "Bearer token",
            "POST /auth/token",
            "client_id and client_secret",
            "1 hour expiry",
        ],
        max_output_tokens=256,
    ),
    RAGFixture(
        id="rag-12",
        complexity=TaskComplexity.MODERATE,
        question=(
            "What was the architectural decision for choosing event sourcing "
            "over CRUD, and what trade-offs were considered?"
        ),
        context_chunks=[
            RAGChunk(
                content=(
                    "ADR-007: Event Sourcing for Order Service\n"
                    "Decision: Use event sourcing with CQRS for the order service.\n"
                    "Context: Order lifecycle requires full audit trail, undo "
                    "capability, and real-time projections for different consumers.\n"
                    "Trade-offs: Higher complexity in event schema evolution, "
                    "eventual consistency between read and write models, "
                    "need for snapshot strategy at >10K events per aggregate.\n"
                    "Rejected: Traditional CRUD — insufficient audit trail, "
                    "no temporal queries."
                ),
                label="relevant",
                source="adr/007-event-sourcing.md",
            ),
            RAGChunk(
                content=(
                    "ADR-003: Use PostgreSQL for primary datastore.\n"
                    "Decision: PostgreSQL 16 with pgvector extension.\n"
                    "Context: Team familiarity, JSON support, vector search."
                ),
                label="distractor",
                source="adr/003-postgresql.md",
            ),
        ],
        gold_criteria=[
            "event sourcing with CQRS",
            "audit trail",
            "schema evolution complexity",
            "eventual consistency",
            "CRUD rejected",
        ],
        max_output_tokens=512,
    ),
    RAGFixture(
        id="rag-13",
        complexity=TaskComplexity.LIGHT,
        question="What is the procedure to roll back a failed Kubernetes deployment?",
        context_chunks=[
            RAGChunk(
                content=(
                    "Runbook: K8s Deployment Rollback\n"
                    "1. Check rollout status: kubectl rollout status deployment/<name>\n"
                    "2. View history: kubectl rollout history deployment/<name>\n"
                    "3. Roll back: kubectl rollout undo deployment/<name>\n"
                    "4. To specific revision: kubectl rollout undo deployment/<name> --to-revision=N\n"
                    "5. Verify: kubectl get pods -w"
                ),
                label="relevant",
                source="runbook-k8s-rollback.md",
            ),
        ],
        gold_criteria=[
            "kubectl rollout undo",
            "rollout history",
            "specific revision",
            "verify with get pods",
        ],
        max_output_tokens=300,
    ),
    RAGFixture(
        id="rag-14",
        complexity=TaskComplexity.MODERATE,
        question="What's causing the 503 errors in the API gateway based on these logs?",
        context_chunks=[
            RAGChunk(
                content=(
                    "API Gateway error log (2026-06-10 14:00-14:30):\n"
                    "14:02 upstream_connect_error: connection refused to backend-svc:8080\n"
                    "14:05 upstream_connect_error: connection refused to backend-svc:8080\n"
                    "14:10 upstream_connect_error: no healthy upstream\n"
                    "14:15 circuit_breaker_open: backend-svc (5 consecutive failures)\n"
                    "14:20 upstream_connect_error: no healthy upstream"
                ),
                label="relevant",
                source="gateway-errors.log",
            ),
            RAGChunk(
                content=(
                    "K8s events (backend-svc):\n"
                    "14:01 Pod backend-svc-7f8d9 OOMKilled (memory limit 512Mi exceeded)\n"
                    "14:01 ReplicaSet scaling down to 0/3 ready\n"
                    "14:03 Pod backend-svc-a2c4f CrashLoopBackOff"
                ),
                label="relevant",
                source="k8s-events.log",
            ),
            RAGChunk(
                content=(
                    "Prometheus alert: disk usage on monitoring-node at 78%. "
                    "Threshold: 85%. Status: warning."
                ),
                label="distractor",
                source="prometheus-alerts.log",
            ),
        ],
        gold_criteria=[
            "OOMKilled",
            "memory limit exceeded",
            "backend pods crashed",
            "circuit breaker opened",
            "no healthy upstream",
        ],
        max_output_tokens=512,
    ),
    RAGFixture(
        id="rag-15",
        complexity=TaskComplexity.LIGHT,
        question="Does the API support batch operations for user creation?",
        context_chunks=[
            RAGChunk(
                content=(
                    "API Reference: User Management\n"
                    "POST /users — Create a single user. Body: {name, email, role}.\n"
                    "GET /users — List users with pagination.\n"
                    "GET /users/:id — Get user by ID.\n"
                    "PATCH /users/:id — Update user fields.\n"
                    "DELETE /users/:id — Deactivate user."
                ),
                label="relevant",
                source="api-docs/users.md",
            ),
        ],
        gold_criteria=[
            "no batch endpoint",
            "not documented",
            "only single user creation",
        ],
        context_contains_answer=False,
        max_output_tokens=200,
    ),
    RAGFixture(
        id="rag-16",
        complexity=TaskComplexity.MODERATE,
        question="What caused the 502 errors during the migration window according to these docs?",
        context_chunks=[
            RAGChunk(
                content=(
                    "Migration Playbook v2.1: Database schema migration from v3 to v4.\n"
                    "Step 3: Run ALTER TABLE on users table (adds 'preferences' JSONB column).\n"
                    "WARNING: This migration takes a full table lock on PostgreSQL < 16. "
                    "Expected duration: 2-5 minutes for tables with >1M rows. "
                    "During this window, all queries to the users table will block."
                ),
                label="relevant",
                source="migration-playbook-v2.1.md",
            ),
            RAGChunk(
                content=(
                    "Application config: connection pool size = 20, "
                    "connection timeout = 5000ms. When all pool connections are "
                    "blocked waiting on table lock, new requests receive 502."
                ),
                label="relevant",
                source="app-config-notes.md",
            ),
            RAGChunk(
                content=(
                    "Q1 OKRs: Reduce p95 latency by 20%. Migrate to arm64 instances. "
                    "Implement canary deployments."
                ),
                label="distractor",
                source="okrs-q1.md",
            ),
        ],
        gold_criteria=[
            "full table lock",
            "ALTER TABLE",
            "connection pool exhaustion",
            "blocked queries",
            "5000ms timeout",
        ],
        max_output_tokens=512,
    ),
    RAGFixture(
        id="rag-17",
        complexity=TaskComplexity.HEAVY,
        question="Create a capacity planning recommendation based on these metrics.",
        context_chunks=[
            RAGChunk(
                content=(
                    "Performance metrics (last 30 days):\n"
                    "- Avg requests/sec: 850 (peak: 2,400 during sale events)\n"
                    "- P50 latency: 45ms, P95: 320ms, P99: 1200ms\n"
                    "- CPU utilization: avg 55%, peak 92%\n"
                    "- Memory: avg 68%, peak 85%\n"
                    "- Current capacity: 6x c5.2xlarge (8 vCPU, 16 GB each)"
                ),
                label="relevant",
                source="metrics-monthly.md",
            ),
            RAGChunk(
                content=(
                    "Growth forecast: 40% traffic increase expected in Q4 due to "
                    "product launch. Marketing projects 3x spike during launch week."
                ),
                label="relevant",
                source="growth-forecast.md",
            ),
            RAGChunk(
                content=(
                    "Cost report: current monthly spend on EC2: $4,200. "
                    "Reserved instance coverage: 60%."
                ),
                label="relevant",
                source="cost-report.md",
            ),
        ],
        gold_criteria=[
            "current peak nearly saturates CPU",
            "Q4 growth requires scaling",
            "3x spike exceeds current capacity",
            "specific instance recommendation",
            "cost implications",
        ],
        max_output_tokens=800,
    ),
    RAGFixture(
        id="rag-18",
        complexity=TaskComplexity.LIGHT,
        question="What version of the API introduced the webhooks feature?",
        context_chunks=[
            RAGChunk(
                content=(
                    "API Changelog:\n"
                    "v2.0 (2025-01-15): Added pagination, rate limiting.\n"
                    "v2.1 (2025-06-01): Added bulk export endpoint.\n"
                    "v2.2 (2025-09-15): Added filtering by date range.\n"
                    "v3.0 (2026-01-01): Breaking: new auth flow, deprecated v1 endpoints."
                ),
                label="relevant",
                source="api-changelog.md",
            ),
        ],
        gold_criteria=[
            "not mentioned in changelog",
            "webhooks not documented",
        ],
        context_contains_answer=False,
        max_output_tokens=200,
    ),
    RAGFixture(
        id="rag-19",
        complexity=TaskComplexity.MODERATE,
        question=(
            "What is the correct procedure for upgrading the cluster from "
            "Kubernetes 1.28 to 1.30 according to the docs?"
        ),
        context_chunks=[
            RAGChunk(
                content=(
                    "K8s Upgrade Guide:\n"
                    "- Always upgrade one minor version at a time (1.28→1.29→1.30).\n"
                    "- Before each upgrade: check API deprecations with kubent.\n"
                    "- Drain nodes one at a time: kubectl drain <node> --ignore-daemonsets.\n"
                    "- Upgrade control plane first, then worker nodes.\n"
                    "- Verify: kubectl get nodes (all should show new version).\n"
                    "- Run conformance tests after upgrade."
                ),
                label="relevant",
                source="k8s-upgrade-guide.md",
            ),
            RAGChunk(
                content=(
                    "Helm chart versions: ingress-nginx 4.8.0 (compatible with "
                    "k8s 1.25-1.30), cert-manager 1.14 (compatible with k8s 1.26-1.30)."
                ),
                label="relevant",
                source="helm-compat-matrix.md",
            ),
        ],
        gold_criteria=[
            "one minor version at a time",
            "check API deprecations",
            "drain nodes",
            "control plane first",
            "conformance tests",
        ],
        max_output_tokens=512,
    ),
    RAGFixture(
        id="rag-20",
        complexity=TaskComplexity.HEAVY,
        question=(
            "Based on these deployment docs, design a zero-downtime deployment "
            "strategy for the payment service."
        ),
        context_chunks=[
            RAGChunk(
                content=(
                    "Payment Service Architecture:\n"
                    "- Stateless REST API behind an ALB\n"
                    "- PostgreSQL with read replicas\n"
                    "- Redis session cache (30s TTL)\n"
                    "- Processing latency SLA: < 500ms p99\n"
                    "- Zero tolerance for double-charging"
                ),
                label="relevant",
                source="payment-architecture.md",
            ),
            RAGChunk(
                content=(
                    "Current deployment: blue/green with manual cutover. "
                    "Rollback time: ~5 minutes. Last incident: stale cache "
                    "caused duplicate charges during v2.3 rollout (2026-02-10)."
                ),
                label="relevant",
                source="deployment-history.md",
            ),
            RAGChunk(
                content=(
                    "Monitoring: Datadog APM traces all payment flows. "
                    "Alert: p99 > 500ms for 5 minutes. PagerDuty escalation."
                ),
                label="relevant",
                source="monitoring-config.md",
            ),
            RAGChunk(
                content=(
                    "Team org chart: 4 backend engineers, 1 SRE, 1 QA. "
                    "Sprint cadence: 2 weeks. Releases every Thursday."
                ),
                label="distractor",
                source="team-info.md",
            ),
        ],
        gold_criteria=[
            "canary or progressive rollout",
            "idempotency for double-charge prevention",
            "cache invalidation strategy",
            "health check before traffic shift",
            "rollback plan",
            "monitoring during rollout",
        ],
        max_output_tokens=1024,
    ),
]

ALL_RAG_FIXTURES = RAG_FIXTURES
