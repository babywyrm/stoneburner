import hashlib
from pathlib import Path

from atomics.archreview.models import TierConfig
from atomics.archreview.pack import build_pack, estimate_tokens


def _mkrepo(tmp_path: Path) -> Path:
    (tmp_path / "routes").mkdir()
    (tmp_path / "routes" / "login.ts").write_text("// auth route\nlogin();\n")
    (tmp_path / "routes" / "search.ts").write_text("// search\nquery(req.q);\n")
    (tmp_path / "package.json").write_text('{"name":"demo"}\n')
    (tmp_path / "README.md").write_text("# demo\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("x\n")
    return tmp_path


def test_pack_is_deterministic(tmp_path):
    repo = _mkrepo(tmp_path)
    cfg = TierConfig(budget_tokens=4000, exclude=("node_modules/**",))
    a = build_pack(repo, cfg)
    b = build_pack(repo, cfg)
    assert a.text == b.text
    assert a.content_hash == b.content_hash
    assert a.content_hash == hashlib.sha256(a.text.encode()).hexdigest()


def test_pack_excludes_globs(tmp_path):
    repo = _mkrepo(tmp_path)
    cfg = TierConfig(budget_tokens=4000, exclude=("node_modules/**",))
    pack = build_pack(repo, cfg)
    assert "node_modules" not in pack.text
    assert "login.ts" in pack.text


def test_pack_respects_token_budget(tmp_path):
    repo = _mkrepo(tmp_path)
    cfg = TierConfig(budget_tokens=5, exclude=("node_modules/**",))  # tiny
    pack = build_pack(repo, cfg)
    assert estimate_tokens(pack.text) <= 5 + 50  # budget + tree/marker headroom
    assert "TRUNCATED" in pack.text


def test_priority_files_come_first(tmp_path):
    repo = _mkrepo(tmp_path)
    cfg = TierConfig(budget_tokens=4000, priority=("routes/login.ts",),
                     exclude=("node_modules/**",))
    pack = build_pack(repo, cfg)
    assert pack.text.index("login.ts") < pack.text.index("search.ts")
