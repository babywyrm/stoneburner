import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

try:
    import sentence_transformers
    import sqlite_vec
except ImportError:
    sqlite_vec = None
    sentence_transformers = None

_RAG_EXTRAS_MISSING = sqlite_vec is None or sentence_transformers is None

from atomics.cli import cli


def test_rag_cli_has_index_options():
    runner = CliRunner()
    result = runner.invoke(cli, ["rag", "--help"])
    assert result.exit_code == 0
    assert "--index" in result.output
    assert "--top-k" in result.output


@pytest.mark.skipif(_RAG_EXTRAS_MISSING, reason="rag extras not installed")
def test_rag_index_builds_database(tmp_path: Path):
    (tmp_path / "doc.txt").write_text("hello world, this is a test document")
    runner = CliRunner()
    db_path = tmp_path / "index.vec"
    result = runner.invoke(cli, ["rag-index", str(tmp_path), "--db", str(db_path)])
    assert result.exit_code == 0, result.output
    assert db_path.exists()
    assert "chunks" in result.output.lower() or "stored" in result.output.lower()


def test_rag_retrieval_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["rag-retrieval", "--help"])
    assert result.exit_code == 0
    assert "--index" in result.output
    assert "--gold" in result.output


@pytest.mark.skipif(_RAG_EXTRAS_MISSING, reason="rag extras not installed")
def test_rag_retrieval_empty_gold_exits_cleanly(tmp_path: Path):
    (tmp_path / "doc.txt").write_text("hello world, this is a test document")
    runner = CliRunner()
    db_path = tmp_path / "index.vec"
    index_result = runner.invoke(cli, ["rag-index", str(tmp_path), "--db", str(db_path)])
    assert index_result.exit_code == 0, index_result.output

    gold_path = tmp_path / "gold.json"
    gold_path.write_text("{}")
    result = runner.invoke(
        cli,
        ["rag-retrieval", "--index", str(db_path), "--gold", str(gold_path)],
    )
    assert result.exit_code == 0, result.output
    assert "No queries in gold file" in result.output
    assert "0.000" in result.output
    assert "MRR: 0.000" in result.output


def _mock_summary():
    summary = MagicMock()
    summary.overall_rag_score = 0.85
    summary.grounding_score = 0.8
    summary.faithfulness_score = 0.75
    summary.abstention_accuracy = 0.9
    summary.hallucination_rate = 0.05
    summary.avg_latency_ms = 120.0
    summary.total_tokens = 200
    summary.total_cost_usd = 0.02
    summary.fixture_results = []
    summary.parse_failure_rate = 0.0
    summary.to_dict.return_value = {"overall_rag_score": 0.85}
    return summary


def _mock_provider():
    provider = AsyncMock()
    provider.name = "mock"
    provider.generate = AsyncMock(
        return_value=SimpleNamespace(
            text="test answer", model="mock-model", input_tokens=5,
            output_tokens=5, total_tokens=10, latency_ms=1.0,
            estimated_cost_usd=0.0,
        )
    )
    return provider


def _mock_repo():
    repo = MagicMock()
    repo.create_run = MagicMock()
    repo.save_task_result = MagicMock()
    repo.complete_run = MagicMock()
    repo.close = MagicMock()
    return repo


def test_rag_cli_runs_with_mocked_provider():
    runner = CliRunner()
    with patch("atomics.commands.rag._make_provider", return_value=_mock_provider()) as make_provider,          patch("atomics.eval.rag.runner.run_rag", new=AsyncMock(return_value=_mock_summary())) as run_rag,          patch("atomics.storage.repository.MetricsRepository", return_value=_mock_repo()) as repo_cls:
        result = runner.invoke(cli, ["rag", "--provider", "ollama", "--judge-provider", "ollama", "--no-save"])
        assert result.exit_code == 0, result.output
        assert "RAG Evaluation" in result.output
        assert "Overall RAG Score" in result.output
        make_provider.assert_called()
        run_rag.assert_called_once()
        repo_cls.assert_not_called()


def test_rag_cli_with_fixtures_filter():
    runner = CliRunner()
    summary = _mock_summary()
    with patch("atomics.commands.rag._make_provider", return_value=_mock_provider()),          patch("atomics.eval.rag.runner.run_rag", new=AsyncMock(return_value=summary)) as run_rag:
        result = runner.invoke(cli, ["rag", "--fixtures", "rag-01,rag-02", "--no-save"])
        assert result.exit_code == 0, result.output
        call_kwargs = run_rag.call_args.kwargs
        assert len(call_kwargs["fixtures"]) == 2
        assert call_kwargs["fixtures"][0].id == "rag-01"


def test_rag_cli_with_invalid_fixtures_exits():
    runner = CliRunner()
    result = runner.invoke(cli, ["rag", "--fixtures", "rag-999", "--no-save"])
    assert result.exit_code == 1
    assert "Unknown fixture IDs" in result.output


def test_rag_cli_with_json_out():
    runner = CliRunner()
    summary = _mock_summary()
    with runner.isolated_filesystem() as fs:
        out_path = Path(fs) / "rag.json"
        with patch("atomics.commands.rag._make_provider", return_value=_mock_provider()),              patch("atomics.eval.rag.runner.run_rag", new=AsyncMock(return_value=summary)),              patch("atomics.storage.repository.MetricsRepository", return_value=_mock_repo()):
            result = runner.invoke(cli, ["rag", "--no-save", "--json-out", str(out_path)])
            assert result.exit_code == 0, result.output
            assert out_path.exists()


def test_rag_cli_with_index_mocks_extras():
    """Run `atomics rag --index` with mocked sentence_transformers/sqlite_vec."""
    runner = CliRunner()
    summary = _mock_summary()
    fake_index = MagicMock()
    fake_index.search = MagicMock(return_value=[])
    fake_index.info = MagicMock(return_value={"chunk_count": "0", "embedding_model": "mock"})

    fake_embedder = MagicMock()
    fake_embedder_class = MagicMock(return_value=fake_embedder)
    fake_index_class = MagicMock(return_value=fake_index)

    fake_modules = {
        "sentence_transformers": MagicMock(),
        "sqlite_vec": MagicMock(),
    }
    with runner.isolated_filesystem() as fs:
        index_path = Path(fs) / "index.vec"
        index_path.write_text("")  # click.Path requires exists=True
        with patch.dict("sys.modules", fake_modules),              patch("atomics.eval.rag.retrieval.LocalSentenceTransformerEmbedder", fake_embedder_class),              patch("atomics.eval.rag.retrieval.RAGIndex", fake_index_class),              patch("atomics.eval.rag.runner.run_rag", new=AsyncMock(return_value=summary)) as run_rag,              patch("atomics.commands.rag._make_provider", return_value=_mock_provider()):
            result = runner.invoke(cli, ["rag", "--index", str(index_path), "--top-k", "3", "--no-save"])
            assert result.exit_code == 0, result.output
            assert run_rag.call_args.kwargs["index"] is fake_index
            assert run_rag.call_args.kwargs["top_k"] == 3


def test_rag_cli_index_missing_exits_on_missing_extra():
    """When --index is used but extras are missing, CLI exits with a helpful message."""
    runner = CliRunner()
    with runner.isolated_filesystem() as fs:
        index_path = Path(fs) / "index.vec"
        index_path.write_text("")
        with patch.dict("sys.modules", {"sentence_transformers": None, "sqlite_vec": None}):
            result = runner.invoke(cli, ["rag", "--index", str(index_path), "--no-save"])
            assert result.exit_code == 1
            assert "RAG indexing requires" in result.output


@pytest.mark.skipif(_RAG_EXTRAS_MISSING, reason="rag extras not installed")
def test_rag_index_force_rebuild(tmp_path: Path):
    (tmp_path / "doc.txt").write_text("first content")
    runner = CliRunner()
    db_path = tmp_path / "index.vec"
    result = runner.invoke(cli, ["rag-index", str(tmp_path), "--db", str(db_path)])
    assert result.exit_code == 0, result.output
    assert "1 files" in result.output or "1 file" in result.output

    (tmp_path / "doc.txt").write_text("second content")
    result = runner.invoke(cli, ["rag-index", str(tmp_path), "--db", str(db_path), "--force"])
    assert result.exit_code == 0, result.output
    assert "1 files" in result.output or "1 file" in result.output


@pytest.mark.skipif(_RAG_EXTRAS_MISSING, reason="rag extras not installed")
def test_rag_index_empty_directory(tmp_path: Path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    runner = CliRunner()
    db_path = tmp_path / "index.vec"
    result = runner.invoke(cli, ["rag-index", str(empty_dir), "--db", str(db_path)])
    assert result.exit_code == 0, result.output
    assert "0 files" in result.output or "0 chunks" in result.output


@pytest.mark.skipif(_RAG_EXTRAS_MISSING, reason="rag extras not installed")
def test_rag_retrieval_with_queries_and_json_out(tmp_path: Path):
    (tmp_path / "doc.txt").write_text("hello world, this is a test document")
    runner = CliRunner()
    db_path = tmp_path / "index.vec"
    index_result = runner.invoke(cli, ["rag-index", str(tmp_path), "--db", str(db_path)])
    assert index_result.exit_code == 0, index_result.output

    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps({"q1": {"relevant": ["doc.txt"], "scores": {"doc.txt": 1.0}}}))
    queries_path = tmp_path / "queries.json"
    queries_path.write_text(json.dumps({"q1": "hello world"}))
    report_path = tmp_path / "report.json"

    result = runner.invoke(
        cli,
        [
            "rag-retrieval",
            "--index", str(db_path),
            "--gold", str(gold_path),
            "--queries", str(queries_path),
            "--json-out", str(report_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "MRR:" in result.output
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["queries"] == 1
    assert "per_query" in report


def test_rag_index_cli_with_mocked_extras(tmp_path: Path):
    """Run rag-index with mocked [rag] extras so it works offline."""
    runner = CliRunner()
    (tmp_path / "doc.txt").write_text("hello world")
    db_path = tmp_path / "index.vec"

    fake_index = MagicMock()
    fake_index.build = MagicMock(return_value=1)
    fake_index_class = MagicMock(return_value=fake_index)
    fake_embedder_class = MagicMock()

    fake_modules = {
        "sentence_transformers": MagicMock(),
        "sqlite_vec": MagicMock(),
    }
    with patch.dict("sys.modules", fake_modules),          patch("atomics.eval.rag.retrieval.LocalSentenceTransformerEmbedder", fake_embedder_class),          patch("atomics.eval.rag.retrieval.RAGIndex", fake_index_class),          patch("atomics.eval.rag.retrieval.load_documents", return_value=[MagicMock()]):
        result = runner.invoke(cli, ["rag-index", str(tmp_path), "--db", str(db_path)])
        assert result.exit_code == 0, result.output
        assert fake_index.build.called


def test_rag_index_cli_missing_extras_exits(tmp_path: Path):
    runner = CliRunner()
    with patch.dict("sys.modules", {"sentence_transformers": None, "sqlite_vec": None}):
        result = runner.invoke(cli, ["rag-index", str(tmp_path)])
        assert result.exit_code == 1
        assert "RAG indexing requires" in result.output


def test_rag_retrieval_cli_with_mocked_extras(tmp_path: Path):
    """Run rag-retrieval with mocked [rag] extras so it works offline."""
    runner = CliRunner()
    db_path = tmp_path / "index.vec"
    db_path.write_text("")
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps({"q1": {"relevant": ["doc.txt"], "scores": {"doc.txt": 1.0}}}))

    fake_index = MagicMock()
    fake_result = MagicMock()
    fake_result.source = "doc.txt"
    fake_index.search = MagicMock(return_value=[fake_result])
    fake_index.info = MagicMock(return_value={"embedding_model": "mock"})
    fake_index_class = MagicMock(return_value=fake_index)
    fake_embedder_class = MagicMock()

    fake_modules = {
        "sentence_transformers": MagicMock(),
        "sqlite_vec": MagicMock(),
    }
    with patch.dict("sys.modules", fake_modules),          patch("atomics.eval.rag.retrieval.LocalSentenceTransformerEmbedder", fake_embedder_class),          patch("atomics.eval.rag.retrieval.RAGIndex", fake_index_class):
        result = runner.invoke(cli, ["rag-retrieval", "--index", str(db_path), "--gold", str(gold_path)])
        assert result.exit_code == 0, result.output
        assert "MRR:" in result.output


def test_rag_retrieval_cli_missing_extras_exits(tmp_path: Path):
    runner = CliRunner()
    db_path = tmp_path / "index.vec"
    db_path.write_text("")
    gold_path = tmp_path / "gold.json"
    gold_path.write_text("{}")
    with patch.dict("sys.modules", {"sentence_transformers": None, "sqlite_vec": None}):
        result = runner.invoke(cli, ["rag-retrieval", "--index", str(db_path), "--gold", str(gold_path)])
        assert result.exit_code == 1
        assert "RAG retrieval requires" in result.output
