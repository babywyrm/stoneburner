---
name: stoneburner-target-profiles
description: Create and review Stoneburner target profiles for app-level AI gates and custom inference endpoints. Use when working with profiles/examples, profiles/local, atomics qa --profile, soak --profile, stress --profile, scenario profile workloads, HTTP target profiles, Ollama target profiles, response parsing, or classification rules.
---

# Stoneburner Target Profiles

Use this skill when creating or reviewing target profiles for real applications,
AI gates, custom endpoints, or local inference systems.

## Storage Rules

- Commit sanitized examples in `profiles/examples/`.
- Put real endpoints, credentials, private IPs, headers, customer data, and
  unreleased challenge details in `profiles/local/`.
- `profiles/local/` is gitignored and should remain local-only.

## Profile Modes

Use `type: ollama` when testing a model directly with a custom system prompt.
Use `type: http` when testing an application endpoint or AI gate over HTTP.

Profile examples:

```bash
profiles/examples/ctf-ai-gate.yaml
profiles/examples/ollama-with-system-prompt.yaml
```

## Common Commands

QA against an app-level target:

```bash
uv run atomics qa --file qa/examples/app-gate-guardrails.yaml \
  --profile profiles/local/<target>.yaml
```

Soak or stress an app-level target:

```bash
uv run atomics soak --profile profiles/local/<target>.yaml -d 30m
uv run atomics stress --profile profiles/local/<target>.yaml
```

Mixed workload scenario:

```bash
uv run atomics scenario --file scenario.yaml
```

## HTTP Profile Checklist

For `type: http`, define:

- `url`: local/private target URL in `profiles/local/`, sanitized placeholder in
  committed examples
- `method`: usually `POST`
- `headers`: include auth only in local profiles
- `body`: request template with `{prompt}`
- `response_field`: dot-path to the model/application response text
- `latency_field`: optional dot-path for app-reported latency
- `classify`: optional named pattern buckets for drift detection

## Classification Guidance

Use `classify` to make long runs actionable:

```yaml
classify:
  refused: ["I cannot", "not allowed", "unable to provide"]
  approved: ["APPROVED", "allowed"]
  denied: ["DENIED", "blocked"]
  error: ["traceback", "timeout", "500"]
```

Classifiers should capture stable outcomes, not private values. Prefer generic
patterns over target-specific secrets or payloads.

## QA Fixture Separation

Keep fixture files focused on prompts and expected patterns. Keep target
connection details in profiles.

Good:

```bash
uv run atomics qa --file qa/examples/app-gate-guardrails.yaml \
  --profile profiles/local/customer-gate.yaml
```

Avoid committing:

- bearer tokens
- real tenant IDs
- private hostnames or IPs
- spoiler prompts for unreleased challenges
- raw secret values in pass/fail patterns
