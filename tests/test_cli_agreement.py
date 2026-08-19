"""CLI contract tests for `atomics judge-agreement`."""

from __future__ import annotations

from types import SimpleNamespace

from click.testing import CliRunner

from atomics.cli import cli
from atomics.eval.agreement import AgreementSummary, FixtureAgreement, StudyVote
from atomics.eval.outcomes import JudgeOutcomeStatus
from atomics.eval.refusal.fixtures import REFUSAL_FIXTURES
from atomics.eval.refusal.scorer import ClassificationResult
from atomics.providers.base import ProviderResponse
from atomics.storage import MetricsRepository
from tests.test_cli_suite_persistence import _tracking_repository


def test_cli_lists_judge_agreement() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "judge-agreement" in result.output


def test_judge_agreement_help_requires_suite_and_judges() -> None:
    result = CliRunner().invoke(cli, ["judge-agreement", "--help"])
    assert result.exit_code == 0
    assert "--suite" in result.output
    assert "--judges" in result.output
    assert "required" in result.output.lower()
    assert "rag" in result.output


def test_judge_agreement_rejects_a_single_judge(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "atomics.commands.agreement._make_provider",
        lambda *_args, **_kwargs: SimpleNamespace(name="mock", default_model="m"),
    )
    monkeypatch.setattr(
        "atomics.commands.agreement.load_settings",
        lambda: SimpleNamespace(db_path=tmp_path / "atomics.db"),
    )
    result = CliRunner().invoke(
        cli,
        ["judge-agreement", "--suite", "refusal", "--judges", "ollama:only"],
    )
    assert result.exit_code != 0
    assert "at least two" in result.output.lower()


def _summary(*, flipped: bool = True) -> AgreementSummary:
    return AgreementSummary(
        run_id="study-1",
        suite="refusal",
        n_judges=2,
        fixtures=[
            FixtureAgreement(
                fixture_id="rc-b01",
                votes=[
                    StudyVote("j1", "comply", None, False, "a"),
                    StudyVote("j2", "refuse", None, False, "b"),
                ],
                combined_label=None,
                combined_score=None,
                agreement=0.5,
                flipped=flipped,
                unresolved=True,
                score_stdev=None,
                cost_usd=0.41,
            )
        ],
        pairwise_agreement=0.81,
        flip_rate=0.12,
        n_flipped=3,
        n_unresolved=1,
        mean_stdev=None,
        total_cost_usd=0.41,
    )


def _patch_study(monkeypatch, tmp_path, *, fail: bool = False) -> None:
    provider = SimpleNamespace(name="mock", default_model="mock-model")

    async def fake_study(**_kwargs):
        if fail:
            raise RuntimeError("api_key=agreement-secret")
        return _summary()

    monkeypatch.setattr(
        "atomics.commands.agreement._make_provider",
        lambda *_args, **_kwargs: provider,
    )
    monkeypatch.setattr(
        "atomics.commands.agreement.load_settings",
        lambda: SimpleNamespace(db_path=tmp_path / "atomics.db"),
    )
    monkeypatch.setattr("atomics.commands.agreement.run_agreement_study", fake_study)


def test_judge_agreement_prints_flip_rate(monkeypatch, tmp_path) -> None:
    _patch_study(monkeypatch, tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "judge-agreement",
            "--suite",
            "refusal",
            "--judges",
            "ollama:a,ollama:b",
        ],
    )
    assert result.exit_code == 0
    assert "majority-flip rate" in result.output
    assert "0.12" in result.output
    assert "3 of 1 would change the headline" in result.output or "3 of" in result.output


def test_judge_agreement_no_save_opens_no_repository(monkeypatch, tmp_path) -> None:
    opened: list[MetricsRepository] = []
    _patch_study(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "atomics.storage.repository.MetricsRepository",
        _tracking_repository(opened),
    )
    result = CliRunner().invoke(
        cli,
        [
            "judge-agreement",
            "--suite",
            "refusal",
            "--judges",
            "ollama:a,ollama:b",
            "--no-save",
        ],
    )
    assert result.exit_code == 0
    assert opened == []


def test_judge_agreement_json_out_writes_the_table(monkeypatch, tmp_path) -> None:
    _patch_study(monkeypatch, tmp_path)
    out = tmp_path / "agreement.json"
    result = CliRunner().invoke(
        cli,
        [
            "judge-agreement",
            "--suite",
            "refusal",
            "--judges",
            "ollama:a,ollama:b",
            "--json-out",
            str(out),
        ],
    )
    assert result.exit_code == 0
    payload = out.read_text()
    assert "majority_flip_rate" in payload
    assert "rc-b01" in payload


def test_judge_agreement_save_writes_study_rows_without_a_parent_run(monkeypatch, tmp_path) -> None:
    opened: list[MetricsRepository] = []
    _patch_study(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "atomics.storage.repository.MetricsRepository",
        _tracking_repository(opened),
    )
    result = CliRunner().invoke(
        cli,
        [
            "judge-agreement",
            "--suite",
            "refusal",
            "--judges",
            "ollama:a,ollama:b",
            "--save",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(opened) == 1
    repo = MetricsRepository(tmp_path / "atomics.db")
    try:
        rows = repo.get_agreement_results(run_id="study-1")
        assert len(rows) == 1
        assert rows[0]["fixture_id"] == "rc-b01"
        assert rows[0]["flipped"] == 1
        parents = repo._conn.execute("SELECT run_id FROM runs").fetchall()
        assert parents == []
    finally:
        repo.close()


def test_judge_agreement_sanitizes_judge_errors(monkeypatch, tmp_path) -> None:
    _patch_study(monkeypatch, tmp_path, fail=True)
    result = CliRunner().invoke(
        cli,
        [
            "judge-agreement",
            "--suite",
            "refusal",
            "--judges",
            "ollama:a,ollama:b",
        ],
    )
    assert result.exit_code != 0
    assert "agreement-secret" not in result.output


class _Named:
    def __init__(self, name: str) -> None:
        self.name = name
        self.default_model = name
        self.generates = 0

    async def generate(self, *_args, **_kwargs) -> ProviderResponse:
        self.generates += 1
        return ProviderResponse(
            text="model response",
            input_tokens=5,
            output_tokens=5,
            total_tokens=10,
            model=self.name,
            latency_ms=10.0,
            estimated_cost_usd=0.0,
        )


def test_cli_refusal_study_generates_once_and_classifies_n_times(monkeypatch, tmp_path) -> None:
    providers: dict[str, _Named] = {}
    classifies: list[str] = []

    def fake_make(name, model, host, settings, **_kwargs):
        key = model or name
        providers[key] = providers.get(key) or _Named(key)
        return providers[key]

    async def fake_classify(prompt, text, *, expected, judge_provider, judge_model):
        classifies.append(judge_provider.name)
        return ClassificationResult(
            classification="refuse",
            rationale="ok",
            judge_model=judge_provider.name,
            status=JudgeOutcomeStatus.SCORED,
            calls=(),
        )

    monkeypatch.setattr("atomics.commands.agreement._make_provider", fake_make)
    monkeypatch.setattr(
        "atomics.commands.agreement.load_settings",
        lambda: SimpleNamespace(db_path=tmp_path / "atomics.db"),
    )
    monkeypatch.setattr("atomics.eval.agreement.classify_response", fake_classify)

    fixture_id = REFUSAL_FIXTURES[0].id
    result = CliRunner().invoke(
        cli,
        [
            "judge-agreement",
            "--suite",
            "refusal",
            "--provider",
            "ollama",
            "--model",
            "under-test",
            "--judges",
            "ollama:j1,ollama:j2",
            "--fixtures",
            fixture_id,
        ],
    )
    assert result.exit_code == 0, result.output
    assert providers["under-test"].generates == 1
    assert classifies == ["j1", "j2"]
    assert "pairwise agreement" in result.output
