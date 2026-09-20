"""Gap report: catalog sizes versus tags already on a host."""

from __future__ import annotations

from datetime import date

from scripts.model_gaps import Gap, Remote, gaps, hf_name_and_size, parse_ollama_library

_CARD = """
<li class="flex">
  <a href="/library/gemma4">
    <span class="bg-[#ddf4ff] px-2">12b</span>
    <span class="bg-[#ddf4ff] px-2">26b</span>
    <span title="Aug 31, 2026 8:52 PM UTC">2 weeks ago</span>
    thinking
  </a>
</li>
<li class="flex">
  <a href="/library/llama3.2">
    <span class="bg-[#ddf4ff] px-2">1b</span>
    <span title="Jan 2, 2025 1:00 AM UTC">1 year ago</span>
  </a>
</li>
"""


def test_parse_keeps_recent_sizes_and_drops_old_cards() -> None:
    found = parse_ollama_library(_CARD, today=date(2026, 9, 20), days=45)
    assert len(found) == 1
    assert found[0].name == "gemma4"
    assert found[0].sizes == ("12b", "26b")
    assert found[0].updated == date(2026, 8, 31)
    assert found[0].thinking is True


def test_hf_name_splits_family_and_size() -> None:
    assert hf_name_and_size("ibm-granite/granite-4.2-30b") == ("granite4.2", "30b")
    assert hf_name_and_size("Qwen/Qwen3.5-9B") == ("qwen3.5", "9b")
    assert hf_name_and_size("ibm-granite/granite-4.2-8b-NVFP4") == ("granite4.2", "8b")


def test_gaps_split_nowhere_from_partial() -> None:
    installed = {
        "laptop": {"gemma4:12b"},
        "beefy": {"gemma4:12b", "granite4.2:8b"},
    }
    remotes = [
        Remote("ollama", "gemma4", ("12b", "26b"), date(2026, 8, 31), True),
        Remote("huggingface", "granite4.2", ("30b",), date(2026, 9, 4), True),
        Remote("huggingface", "qwen3.5", ("9b",), date(2026, 9, 1), True),
    ]
    installed["laptop"].add("qwen3.5:9b")
    installed["beefy"].add("qwen3.5:9b")
    rows = gaps(installed, remotes)
    by_size = {(row.name, row.size): row for row in rows}
    assert ("qwen3.5", "9b") not in by_size
    assert ("gemma4", "12b") not in by_size
    assert by_size[("gemma4", "26b")] == Gap(
        "ollama",
        "gemma4",
        "26b",
        "2026-08-31",
        ("laptop", "beefy"),
        "known",
    )
    assert by_size[("granite4.2", "30b")].registry == "known"
    assert by_size[("granite4.2", "30b")].missing_on == ("laptop", "beefy")
