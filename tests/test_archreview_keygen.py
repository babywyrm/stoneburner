import textwrap

from atomics.archreview.keygen import answer_key_from_challenges, load_repo_spec


def test_answer_key_from_challenges(tmp_path):
    challenges = textwrap.dedent("""\
        - name: A
          category: Broken Access Control
          difficulty: 3
        - name: B
          category: Broken Access Control
          difficulty: 1
        - name: C
          category: XSS
          difficulty: 2
        - name: D
          category: Totally Unknown Category
          difficulty: 5
    """)
    f = tmp_path / "challenges.yml"
    f.write_text(challenges)
    ak = answer_key_from_challenges(f)
    # Broken Access Control: 3+1 = 4 ; XSS: 2 ; unknown category dropped.
    assert ak.weights["broken_access_control"] == 4.0
    assert ak.weights["xss"] == 2.0
    assert "miscellaneous" not in ak.weights or "unknown" not in ak.weights


def test_load_repo_spec(tmp_path, monkeypatch):
    spec_yaml = textwrap.dedent("""\
        repo:
          name: demo
          git_ref: abc123
          path_env: DEMO_PATH
        tiers:
          floor:
            budget_tokens: 16000
            priority: ["routes/**"]
            exclude: ["node_modules/**"]
          wide:
            budget_tokens: 48000
            priority: ["routes/**", "lib/**"]
            exclude: ["node_modules/**"]
          local:
            budget_tokens: 32000
            priority: ["routes/**", "lib/**"]
            exclude: ["node_modules/**"]
        answer_key:
          version: 1
          categories:
            - {id: injection, weight: 7.5}
            - {id: xss, weight: 6.0}
    """)
    p = tmp_path / "demo.yaml"
    p.write_text(spec_yaml)
    spec = load_repo_spec(p)
    assert spec.name == "demo"
    assert spec.tier("floor").budget_tokens == 16000
    assert spec.tier("local").budget_tokens == 32000
    assert spec.tier("wide").budget_tokens == 48000
    assert spec.answer_key.weights["injection"] == 7.5
