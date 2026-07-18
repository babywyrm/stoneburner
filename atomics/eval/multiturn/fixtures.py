"""Multi-turn conversation fixtures — 15 conversations testing context retention and coherence."""

from __future__ import annotations

from atomics.eval.multiturn import ConversationFixture, ConversationTurn
from atomics.models import TaskComplexity

MULTITURN_FIXTURES: list[ConversationFixture] = [
    # ── CONTEXT RETENTION ─────────────────────────────────────────────────────
    ConversationFixture(
        id="mt-eval-01",
        complexity=TaskComplexity.LIGHT,
        system_prompt="You are a helpful technical assistant.",
        turns=[
            ConversationTurn(
                user_message="My name is Alex and I'm working on a Python web app using FastAPI.",
                expected_behavior="Acknowledge the name and tech stack.",
                gold_criteria=["acknowledge Alex", "note FastAPI"],
            ),
            ConversationTurn(
                user_message="What testing framework would you recommend for my project?",
                expected_behavior="Recommend pytest (standard for Python/FastAPI) and reference the user's stack.",
                gold_criteria=["pytest", "reference FastAPI or Python"],
            ),
            ConversationTurn(
                user_message="Thanks! Can you remind me what framework I said I was using?",
                expected_behavior="Correctly recall FastAPI from turn 1.",
                gold_criteria=["FastAPI"],
            ),
        ],
        conversation_criteria=[
            "remembers user's name (Alex) throughout",
            "remembers the tech stack (FastAPI) when asked to recall",
            "testing recommendation is appropriate for the stated stack",
        ],
    ),
    ConversationFixture(
        id="mt-eval-02",
        complexity=TaskComplexity.MODERATE,
        system_prompt="You are a database consultant helping design a schema.",
        turns=[
            ConversationTurn(
                user_message="We need to store user profiles with name, email, and preferences as JSON. We're using PostgreSQL.",
                expected_behavior="Propose a schema using JSONB for preferences.",
                gold_criteria=["JSONB", "users table"],
            ),
            ConversationTurn(
                user_message="Good. Now we also need an orders table that references users. Each order has items as a nested structure.",
                expected_behavior="Add orders table with FK to users, use JSONB or separate items table.",
                gold_criteria=["foreign key", "orders table", "items"],
            ),
            ConversationTurn(
                user_message="What indexes should I add across both tables for common query patterns?",
                expected_behavior="Recommend indexes considering both tables and their relationship.",
                gold_criteria=["index on email", "index on user_id FK", "GIN index for JSONB"],
            ),
        ],
        conversation_criteria=[
            "schema design is consistent across all turns",
            "later recommendations reference earlier design decisions",
            "no contradictions in data types or relationships",
        ],
    ),
    ConversationFixture(
        id="mt-eval-03",
        complexity=TaskComplexity.LIGHT,
        system_prompt="You are a helpful assistant.",
        turns=[
            ConversationTurn(
                user_message="I have three servers: web-01, web-02, and db-01. web-01 runs nginx, web-02 runs Apache, db-01 runs PostgreSQL.",
                expected_behavior="Acknowledge the infrastructure setup.",
                gold_criteria=["web-01 nginx", "web-02 Apache", "db-01 PostgreSQL"],
            ),
            ConversationTurn(
                user_message="Which of my servers is running Apache?",
                expected_behavior="Correctly identify web-02.",
                gold_criteria=["web-02"],
            ),
            ConversationTurn(
                user_message="And what database is on db-01?",
                expected_behavior="Correctly recall PostgreSQL.",
                gold_criteria=["PostgreSQL"],
            ),
        ],
        conversation_criteria=[
            "correctly maps each server to its service from turn 1",
            "no server-service mix-ups across turns",
        ],
    ),
    # ── INSTRUCTION FOLLOWING ─────────────────────────────────────────────────
    ConversationFixture(
        id="mt-eval-04",
        complexity=TaskComplexity.MODERATE,
        system_prompt="You are a concise technical writer.",
        turns=[
            ConversationTurn(
                user_message="From now on, always format your responses as bullet points. No prose paragraphs.",
                expected_behavior="Acknowledge the format constraint.",
                gold_criteria=["acknowledge bullet point format"],
            ),
            ConversationTurn(
                user_message="Explain the benefits of containerization.",
                expected_behavior="List benefits in bullet point format (not prose).",
                gold_criteria=["bullet points", "isolation", "portability"],
            ),
            ConversationTurn(
                user_message="Now explain the drawbacks.",
                expected_behavior="List drawbacks still in bullet point format.",
                gold_criteria=["bullet points", "complexity", "overhead"],
            ),
            ConversationTurn(
                user_message="Summarize both in a comparison.",
                expected_behavior="Comparison still using bullet points.",
                gold_criteria=["bullet points", "benefits vs drawbacks"],
            ),
        ],
        conversation_criteria=[
            "bullet point format maintained in ALL turns after the instruction",
            "no prose paragraphs after the format was set in turn 1",
        ],
    ),
    ConversationFixture(
        id="mt-eval-05",
        complexity=TaskComplexity.LIGHT,
        system_prompt="You are a math tutor.",
        turns=[
            ConversationTurn(
                user_message="Let's define x = 7 and y = 3.",
                expected_behavior="Acknowledge the variable definitions.",
                gold_criteria=["x = 7", "y = 3"],
            ),
            ConversationTurn(
                user_message="What is x + y?",
                expected_behavior="Calculate 10.",
                gold_criteria=["10"],
            ),
            ConversationTurn(
                user_message="Now let y = 5. What is x * y?",
                expected_behavior="Use updated y=5, calculate 35.",
                gold_criteria=["35", "y = 5"],
            ),
            ConversationTurn(
                user_message="What is the current value of y?",
                expected_behavior="Correctly state y = 5 (the updated value).",
                gold_criteria=["5"],
            ),
        ],
        conversation_criteria=[
            "tracks variable updates correctly",
            "uses updated y=5 not original y=3 after the reassignment",
        ],
    ),
    # ── COHERENCE & CONSISTENCY ───────────────────────────────────────────────
    ConversationFixture(
        id="mt-eval-06",
        complexity=TaskComplexity.MODERATE,
        system_prompt="You are a security consultant.",
        turns=[
            ConversationTurn(
                user_message="We're evaluating whether to use JWT or session-based auth. What do you recommend for our REST API?",
                expected_behavior="Give a recommendation with trade-offs of each.",
                gold_criteria=["JWT", "session", "trade-offs"],
            ),
            ConversationTurn(
                user_message="What are the security risks of the approach you recommended?",
                expected_behavior="List risks consistent with the recommendation from turn 1.",
                gold_criteria=["risks specific to recommended approach"],
            ),
            ConversationTurn(
                user_message="We've decided to go with the other approach instead. What do we need to implement?",
                expected_behavior="Correctly identify which approach was NOT recommended and give implementation guidance.",
                gold_criteria=["implementation steps for the alternative"],
            ),
        ],
        conversation_criteria=[
            "maintains a consistent recommendation across turns",
            "correctly tracks which approach was recommended vs alternative",
            "doesn't contradict earlier analysis when discussing the switch",
        ],
    ),
    ConversationFixture(
        id="mt-eval-07",
        complexity=TaskComplexity.HEAVY,
        system_prompt="You are an experienced SRE helping plan a migration.",
        turns=[
            ConversationTurn(
                user_message=(
                    "We're migrating from a monolith to microservices. Our monolith is a Python Django app "
                    "with 50K LOC, PostgreSQL, Redis cache, and Celery workers. Traffic is 500 req/s peak."
                ),
                expected_behavior="Acknowledge the scope and suggest a migration strategy.",
                gold_criteria=["strangler fig or incremental", "service boundaries"],
            ),
            ConversationTurn(
                user_message="Which service should we extract first and why?",
                expected_behavior="Recommend a specific bounded context with reasoning.",
                gold_criteria=["specific service recommendation", "reasoning based on coupling"],
            ),
            ConversationTurn(
                user_message="How do we handle the shared database during the transition?",
                expected_behavior="Discuss database decomposition strategies referencing the specific services mentioned.",
                gold_criteria=["shared DB pattern", "eventual migration", "reference earlier services"],
            ),
            ConversationTurn(
                user_message="What's our rollback plan if the first extracted service fails in production?",
                expected_behavior="Rollback plan specific to the service recommended in turn 2.",
                gold_criteria=["rollback for the specific service", "feature flags or routing"],
            ),
        ],
        conversation_criteria=[
            "consistent migration strategy across all turns",
            "rollback plan references the specific service from turn 2",
            "database advice is consistent with the migration approach from turn 1",
        ],
    ),
    # ── CORRECTION HANDLING ───────────────────────────────────────────────────
    ConversationFixture(
        id="mt-eval-08",
        complexity=TaskComplexity.LIGHT,
        system_prompt="You are a helpful assistant.",
        turns=[
            ConversationTurn(
                user_message="The capital of Australia is Sydney, right?",
                expected_behavior="Politely correct: the capital is Canberra, not Sydney.",
                gold_criteria=["Canberra", "correction"],
            ),
            ConversationTurn(
                user_message="Oh right, thanks. Tell me more about the capital.",
                expected_behavior="Discuss Canberra (not Sydney).",
                gold_criteria=["Canberra", "facts about the capital"],
            ),
        ],
        conversation_criteria=[
            "corrects the misconception in turn 1",
            "consistently discusses Canberra (not Sydney) in turn 2",
        ],
    ),
    ConversationFixture(
        id="mt-eval-09",
        complexity=TaskComplexity.MODERATE,
        system_prompt="You are a code reviewer.",
        turns=[
            ConversationTurn(
                user_message="Review this function:\ndef add(a, b): return a + b",
                expected_behavior="Note it works but suggest type hints, docstring.",
                gold_criteria=["type hints", "docstring"],
            ),
            ConversationTurn(
                user_message="Actually, I realize I pasted the wrong code. Here's the real one:\ndef divide(a, b): return a / b",
                expected_behavior="Review the DIVIDE function (not add), note missing zero-division check.",
                gold_criteria=["divide", "zero division", "not reference add"],
            ),
            ConversationTurn(
                user_message="Can you suggest the corrected version?",
                expected_behavior="Correct the divide function (not add).",
                gold_criteria=["divide", "zero check", "corrected code"],
            ),
        ],
        conversation_criteria=[
            "switches context from add() to divide() cleanly",
            "does not mix up the two functions",
            "corrected version is for divide, not add",
        ],
    ),
    # ── MULTI-STEP REASONING ─────────────────────────────────────────────────
    ConversationFixture(
        id="mt-eval-10",
        complexity=TaskComplexity.HEAVY,
        system_prompt="You are a network security analyst.",
        turns=[
            ConversationTurn(
                user_message=(
                    "We detected traffic from 10.0.1.5 to an external IP 203.0.113.42 "
                    "on port 443. The internal host is in the HR VLAN. Volume: 2GB in 30 minutes."
                ),
                expected_behavior="Flag as suspicious: unusual volume, HR VLAN to external.",
                gold_criteria=["suspicious", "unusual volume", "HR VLAN"],
            ),
            ConversationTurn(
                user_message="I checked and 203.0.113.42 resolves to a known Dropbox IP. Does that change your assessment?",
                expected_behavior="Moderate assessment but still flag the volume as unusual.",
                gold_criteria=["Dropbox is legitimate", "volume still concerning"],
            ),
            ConversationTurn(
                user_message="The user on 10.0.1.5 submitted their resignation yesterday. Update your analysis.",
                expected_behavior="Escalate to likely data exfiltration by departing employee.",
                gold_criteria=["data exfiltration", "departing employee", "escalate"],
            ),
            ConversationTurn(
                user_message="What containment steps should we take right now?",
                expected_behavior="Recommend specific steps referencing the IP, user, and Dropbox.",
                gold_criteria=["disable account", "block 10.0.1.5", "preserve evidence"],
            ),
        ],
        conversation_criteria=[
            "analysis evolves correctly with each new piece of evidence",
            "containment references specific IPs and facts from earlier turns",
            "final assessment correctly integrates all three pieces of evidence",
        ],
    ),
    # ── CONSTRAINT ACCUMULATION ──────────────────────────────────────────────
    ConversationFixture(
        id="mt-eval-11",
        complexity=TaskComplexity.MODERATE,
        system_prompt="You are a travel planning assistant.",
        turns=[
            ConversationTurn(
                user_message="I want to plan a vacation. Budget is $3000 total.",
                expected_behavior="Acknowledge budget constraint.",
                gold_criteria=["$3000 budget"],
            ),
            ConversationTurn(
                user_message="I'd like to go somewhere warm, preferably a beach destination.",
                expected_behavior="Suggest warm beach destinations within budget.",
                gold_criteria=["beach", "within $3000"],
            ),
            ConversationTurn(
                user_message="Oh, I should mention I'm vegetarian and my partner is allergic to seafood.",
                expected_behavior="Acknowledge dietary constraints, adjust recommendations.",
                gold_criteria=["vegetarian", "seafood allergy", "adjust recommendations"],
            ),
            ConversationTurn(
                user_message="Can you give me a final recommendation that accounts for everything we discussed?",
                expected_behavior="Recommendation respects ALL constraints: budget, beach, vegetarian, no seafood.",
                gold_criteria=["$3000", "beach", "vegetarian-friendly", "seafood-free dining"],
            ),
        ],
        conversation_criteria=[
            "final recommendation respects ALL accumulated constraints",
            "no constraint from earlier turns is forgotten",
        ],
    ),
    ConversationFixture(
        id="mt-eval-12",
        complexity=TaskComplexity.LIGHT,
        system_prompt="You are a helpful assistant. Always respond in exactly 3 sentences.",
        turns=[
            ConversationTurn(
                user_message="What is Docker?",
                expected_behavior="Explain Docker in exactly 3 sentences.",
                gold_criteria=["3 sentences", "containerization"],
            ),
            ConversationTurn(
                user_message="How does it differ from VMs?",
                expected_behavior="Explain difference in exactly 3 sentences.",
                gold_criteria=["3 sentences", "lighter than VMs"],
            ),
            ConversationTurn(
                user_message="Give me a use case.",
                expected_behavior="Describe a use case in exactly 3 sentences.",
                gold_criteria=["3 sentences", "practical example"],
            ),
        ],
        conversation_criteria=[
            "system prompt constraint (3 sentences) maintained across ALL turns",
        ],
    ),
    # ── SECURITY-FOCUSED MULTI-TURN ──────────────────────────────────────────
    ConversationFixture(
        id="mt-eval-13",
        complexity=TaskComplexity.HEAVY,
        system_prompt="You are a penetration tester documenting findings for a client report.",
        turns=[
            ConversationTurn(
                user_message="I found an IDOR on /api/users/{id}. I could access other users' data by changing the ID parameter.",
                expected_behavior="Document the IDOR finding with severity and remediation.",
                gold_criteria=["IDOR", "high severity", "authorization check"],
            ),
            ConversationTurn(
                user_message="I also found the API returns full credit card numbers in the response body for /api/users/{id}/payments.",
                expected_behavior="Document PII exposure, note it compounds with the IDOR.",
                gold_criteria=["PII exposure", "compounds with IDOR", "mask card numbers"],
            ),
            ConversationTurn(
                user_message="Write the executive summary for these two findings together.",
                expected_behavior="Summary that ties both findings together as a compound risk.",
                gold_criteria=["compound risk", "IDOR + PII", "executive language"],
            ),
        ],
        conversation_criteria=[
            "executive summary references BOTH findings from earlier turns",
            "correctly identifies the compound risk (IDOR enables PII access)",
        ],
    ),
    ConversationFixture(
        id="mt-eval-14",
        complexity=TaskComplexity.MODERATE,
        system_prompt="You are a DevOps engineer helping configure a CI/CD pipeline.",
        turns=[
            ConversationTurn(
                user_message="We use GitHub Actions. Our app is a Node.js monorepo with packages/ directory containing api, web, and shared.",
                expected_behavior="Acknowledge the setup, suggest monorepo-aware CI patterns.",
                gold_criteria=["monorepo", "path-based triggers", "packages/"],
            ),
            ConversationTurn(
                user_message="We want tests to run only for the package that changed, not all three.",
                expected_behavior="Suggest path-based job triggers filtering by packages/api, packages/web, etc.",
                gold_criteria=["path filter", "dorny/paths-filter or on.push.paths"],
            ),
            ConversationTurn(
                user_message="What about the shared package? If it changes, we need to test everything.",
                expected_behavior="Add a rule: changes to packages/shared trigger all test jobs.",
                gold_criteria=["shared triggers all", "conditional logic"],
            ),
        ],
        conversation_criteria=[
            "CI configuration is consistent and additive across turns",
            "shared package rule doesn't contradict earlier per-package filtering",
        ],
    ),
    ConversationFixture(
        id="mt-eval-15",
        complexity=TaskComplexity.MODERATE,
        system_prompt="You are a Kubernetes administrator.",
        turns=[
            ConversationTurn(
                user_message="Our cluster has 3 namespaces: prod, staging, dev. We want network policies so pods in dev can't talk to prod.",
                expected_behavior="Suggest NetworkPolicy denying dev→prod traffic.",
                gold_criteria=["NetworkPolicy", "deny dev to prod"],
            ),
            ConversationTurn(
                user_message="Staging should be able to talk to prod's database but nothing else in prod.",
                expected_behavior="Refine policy to allow staging→prod-db only.",
                gold_criteria=["staging to prod-db allowed", "other prod denied"],
            ),
            ConversationTurn(
                user_message="Can you write the YAML for both policies?",
                expected_behavior="Two NetworkPolicy YAMLs consistent with all constraints.",
                gold_criteria=["two policies", "dev denied", "staging to prod-db only"],
            ),
        ],
        conversation_criteria=[
            "YAML policies match ALL constraints from both turns",
            "no contradiction between the two policies",
        ],
    ),
]

ALL_MULTITURN_FIXTURES = MULTITURN_FIXTURES
