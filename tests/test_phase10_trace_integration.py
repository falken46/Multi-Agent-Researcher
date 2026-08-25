from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import agents.researcher as researcher_module
import core.llm as llm_module
from agents.graph import create_initial_state, graph
from core.config import clear_settings_cache
from core.trace import summarize
from tools.kb_search import KBSearchResult


def _response(content: str) -> Any:
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


class RoutingSyncCompletions:
    def __init__(self) -> None:
        self.call_count = 0

    def create(self, **kwargs: Any) -> Any:
        self.call_count += 1
        if "response_format" in kwargs:
            return _response('{"sub_questions":["Q1","Q2","Q3"]}')
        return _response("# Test report")


class RoutingAsyncCompletions:
    def __init__(self) -> None:
        self.call_count = 0

    async def create(self, **kwargs: Any) -> Any:
        self.call_count += 1
        if "response_format" in kwargs:
            return _response(
                '{"quality_score":0.9,"critique":"sufficient",'
                '"missing_aspects":[]}'
            )
        user_prompt = str(kwargs["messages"][-1]["content"])
        question = next(
            item for item in ("Q1", "Q2", "Q3") if f"子问题:\n{item}" in user_prompt
        )
        return _response(f"{question} summary")


@pytest.mark.asyncio
async def test_full_graph_writes_one_trace_with_all_llm_calls(
    monkeypatch: pytest.MonkeyPatch,
    runtime_dir: Path,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("TRACE_ENABLED", "true")
    monkeypatch.setenv("TRACE_DIR", str(runtime_dir))
    clear_settings_cache()

    sync_completions = RoutingSyncCompletions()
    async_completions = RoutingAsyncCompletions()
    sync_client = SimpleNamespace(chat=SimpleNamespace(completions=sync_completions))
    async_client = SimpleNamespace(chat=SimpleNamespace(completions=async_completions))
    monkeypatch.setattr(llm_module, "_create_sync_client", lambda settings: sync_client)
    monkeypatch.setattr(llm_module, "_create_async_client", lambda settings: async_client)

    def fake_kb_search(
        query: str,
        top_n: int,
        *,
        trace_id: str | None = None,
    ) -> KBSearchResult:
        return {
            "hits": [
                {
                    "text": f"{query} evidence",
                    "source": f"data/kb/{query}.md",
                    "chunk_index": 0,
                    "score": 0.9,
                    "channel": "vector+bm25",
                }
            ],
            "max_score": 0.9,
        }

    def unexpected_web_search(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("high-score local evidence must not trigger web fallback")

    monkeypatch.setattr(researcher_module, "kb_search", fake_kb_search)
    monkeypatch.setattr(researcher_module, "web_search", unexpected_web_search)
    initial_state = create_initial_state("Test topic")

    final_state = await graph.ainvoke(initial_state)
    summary = summarize(initial_state["trace_id"])

    assert final_state["final_report"] == "# Test report"
    assert sync_completions.call_count == 2
    assert async_completions.call_count == 4
    assert summary["llm_calls"] == 6
    assert summary["total_tokens"] == 72
    assert summary["by_node"]["planner"]["calls"] == 1
    assert summary["by_node"]["researcher"]["calls"] == 3
    assert summary["by_node"]["critic"]["calls"] == 1
    assert summary["by_node"]["writer"]["calls"] == 1
