from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import core.llm as llm_module
import agents.researcher as researcher_module
from agents.graph import create_initial_state, graph
from core.config import clear_settings_cache
from core.trace import summarize
from tools.web_search import SearchResult


class SequentialCompletions:
    def __init__(self, contents: list[str]) -> None:
        self.contents = contents

    def create(self, **kwargs: Any) -> Any:
        content = self.contents.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
                prompt_cache_hit_tokens=0,
                prompt_cache_miss_tokens=10,
            ),
        )


def test_full_graph_writes_one_trace_with_all_llm_calls(
    monkeypatch: pytest.MonkeyPatch,
    runtime_dir: Path,
) -> None:
    trace_dir = runtime_dir
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("TRACE_ENABLED", "true")
    monkeypatch.setenv("TRACE_DIR", str(trace_dir))
    clear_settings_cache()

    completions = SequentialCompletions(
        [
            '{"sub_questions":["Q1","Q2","Q3"]}',
            "Q1 summary",
            "Q2 summary",
            "Q3 summary",
            "# Test report",
        ]
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(llm_module, "_create_sync_client", lambda settings: client)

    def fake_web_search(query: str, max_results: int) -> list[SearchResult]:
        return [
            {
                "title": f"{query} source",
                "url": f"https://example.com/{query}",
                "snippet": f"{query} evidence",
                "source": "tavily",
            }
        ]

    monkeypatch.setattr(researcher_module, "web_search", fake_web_search)
    initial_state = create_initial_state("Test topic")

    final_state = graph.invoke(initial_state)
    summary = summarize(initial_state["trace_id"])

    assert final_state["final_report"] == "# Test report"
    assert completions.contents == []
    assert summary["llm_calls"] == 5
    assert summary["total_tokens"] == 60
    assert summary["by_node"]["planner"]["calls"] == 1
    assert summary["by_node"]["researcher"]["calls"] == 3
    assert summary["by_node"]["writer"]["calls"] == 1
