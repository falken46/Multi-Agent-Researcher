from __future__ import annotations

import asyncio
import importlib
from collections.abc import Sequence

import pytest

from agents.graph import create_initial_state
from agents.state import ResearchState
from core.config import clear_settings_cache
from tools.kb_search import KBSearchResult
from tools.web_search import SearchResult

researcher_module = importlib.import_module("agents.researcher")


def make_state(sub_questions: list[str] | None = None) -> ResearchState:
    state = create_initial_state("AI Agent 趋势")
    state["sub_questions"] = (
        sub_questions if sub_questions is not None else ["问题 1", "问题 2", "问题 3"]
    )
    return state


def make_kb_result(
    question: str,
    *,
    score: float = 0.9,
    fallback_confidence: float | None = None,
) -> KBSearchResult:
    confidence = score if fallback_confidence is None else fallback_confidence
    return {
        "hits": [
            {
                "chunk_id": f"chunk-{question}",
                "text": f"{question} 本地资料",
                "source": f"kb/{question}.md",
                "chunk_index": 0,
                "ranking_score": score,
                "score": score,
                "score_kind": "rrf",
                "fallback_confidence": confidence,
                "channel": "hybrid",
            }
        ],
        "max_score": score,
        "fallback_confidence": confidence,
        "fallback_confidence_kind": "vector_cosine_similarity",
    }


def make_search_result(question: str) -> SearchResult:
    return {
        "title": f"{question} title",
        "url": f"https://example.com/{question}",
        "snippet": f"{question} web snippet",
        "source": "tavily",
    }


@pytest.fixture(autouse=True)
def fake_embedding_backend(monkeypatch: pytest.MonkeyPatch):
    """Phase 12 的 Researcher 测试不得依赖真实 embedding 或网络。"""
    monkeypatch.setenv("EMBEDDING_BACKEND", "fake")
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.mark.asyncio
async def test_researcher_uses_high_confidence_local_evidence_without_web(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kb_queries: list[str] = []
    captured_origins: list[str] = []

    def fake_kb_search(query: str, top_n: int, *, trace_id: str) -> KBSearchResult:
        assert top_n == 5
        assert trace_id
        kb_queries.append(query)
        return make_kb_result(query, score=0.02, fallback_confidence=0.9)

    def unexpected_web_search(query: str, max_results: int) -> list[SearchResult]:
        raise AssertionError(f"高分本地结果不应触发联网检索: {query}")

    async def fake_summary(
        question: str,
        evidence: Sequence[dict[str, str]],
        system_prompt: str,
        trace_id: str,
    ) -> str:
        assert "Researcher Agent" in system_prompt
        assert trace_id
        captured_origins.extend(item["origin"] for item in evidence)
        return f"{question} 的资料摘要"

    monkeypatch.setattr(researcher_module, "kb_search", fake_kb_search)
    monkeypatch.setattr(researcher_module, "web_search", unexpected_web_search)
    monkeypatch.setattr(researcher_module, "_call_summary_model", fake_summary)

    result = await researcher_module.researcher_node(make_state(["问题 A", "问题 B"]))

    assert kb_queries == ["问题 A", "问题 B"]
    assert captured_origins == ["kb", "kb"]
    assert result["errors"] == []
    assert result["fallback_queries"] == []
    assert result["research_results"]["问题 A"].startswith("问题 A 的资料摘要")
    assert "kb/问题 A.md" in result["research_results"]["问题 A"]
    assert result["citations"]["问题 A"] == [
        {
            "source": "kb/问题 A.md",
            "origin": "kb",
            "snippet": "问题 A 本地资料",
        }
    ]


@pytest.mark.asyncio
async def test_researcher_falls_back_to_web_below_stable_confidence_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    web_queries: list[str] = []
    captured_origins: list[str] = []

    def fake_kb_search(query: str, top_n: int, *, trace_id: str) -> KBSearchResult:
        return make_kb_result(query, score=0.95, fallback_confidence=0.1)

    def fake_web_search(query: str, max_results: int) -> list[SearchResult]:
        assert max_results == 3
        web_queries.append(query)
        return [make_search_result(query)]

    async def fake_summary(
        question: str,
        evidence: Sequence[dict[str, str]],
        system_prompt: str,
        trace_id: str,
    ) -> str:
        captured_origins.extend(item["origin"] for item in evidence)
        return f"{question} 混合摘要"

    monkeypatch.setattr(researcher_module, "kb_search", fake_kb_search)
    monkeypatch.setattr(researcher_module, "web_search", fake_web_search)
    monkeypatch.setattr(researcher_module, "_call_summary_model", fake_summary)

    result = await researcher_module.researcher_node(make_state(["低分问题"]))

    assert web_queries == ["低分问题"]
    assert captured_origins == ["kb", "web"]
    assert result["fallback_queries"] == ["低分问题"]
    assert [item["origin"] for item in result["citations"]["低分问题"]] == [
        "kb",
        "web",
    ]


@pytest.mark.asyncio
async def test_researcher_can_fail_fast_on_injected_cache_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FatalCacheError(RuntimeError):
        pass

    def fake_kb_search(query: str, top_n: int, *, trace_id: str) -> KBSearchResult:
        return make_kb_result(query, fallback_confidence=0.0)

    def failed_cached_search(query: str, max_results: int) -> list[SearchResult]:
        raise FatalCacheError(f"cache miss: {query}")

    monkeypatch.setattr(researcher_module, "kb_search", fake_kb_search)

    with pytest.raises(FatalCacheError, match="cache miss"):
        await researcher_module.researcher_node(
            make_state(["必须缓存的问题"]),
            web_searcher=failed_cached_search,
            fail_fast_exceptions=(FatalCacheError,),
        )


@pytest.mark.asyncio
async def test_researcher_honors_configured_concurrency_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCH_CONCURRENCY", "2")
    clear_settings_cache()
    active = 0
    max_active = 0

    def fake_kb_search(query: str, top_n: int, *, trace_id: str) -> KBSearchResult:
        return make_kb_result(query)

    async def slow_summary(
        question: str,
        evidence: Sequence[dict[str, str]],
        system_prompt: str,
        trace_id: str,
    ) -> str:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return f"{question} 摘要"

    monkeypatch.setattr(researcher_module, "kb_search", fake_kb_search)
    monkeypatch.setattr(researcher_module, "_call_summary_model", slow_summary)

    result = await researcher_module.researcher_node(
        make_state(["Q1", "Q2", "Q3", "Q4", "Q5"])
    )

    assert max_active == 2
    assert active == 0
    assert len(result["research_results"]) == 5


@pytest.mark.asyncio
async def test_researcher_parallel_run_is_faster_than_serial_run_using_trace_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一组 fake IO 做 1/3 并发对照，耗时只取 node_end trace payload。"""
    trace_events: list[dict[str, object]] = []

    def fake_kb_search(query: str, top_n: int, *, trace_id: str) -> KBSearchResult:
        return make_kb_result(query)

    async def delayed_summary(
        question: str,
        evidence: Sequence[dict[str, str]],
        system_prompt: str,
        trace_id: str,
    ) -> str:
        await asyncio.sleep(0.05)
        return f"{question} 摘要"

    def capture_trace(event: dict[str, object]) -> None:
        trace_events.append(event)

    monkeypatch.setattr(researcher_module, "kb_search", fake_kb_search)
    monkeypatch.setattr(researcher_module, "_call_summary_model", delayed_summary)
    monkeypatch.setattr(researcher_module, "emit", capture_trace)

    async def run_with_concurrency(value: int) -> float:
        monkeypatch.setenv("RESEARCH_CONCURRENCY", str(value))
        clear_settings_cache()
        trace_events.clear()
        result = await researcher_module.researcher_node(make_state(["Q1", "Q2", "Q3"]))
        assert len(result["research_results"]) == 3
        node_end = [
            event
            for event in trace_events
            if event["event"] == "node_end" and event["node"] == "researcher"
        ]
        assert len(node_end) == 1
        payload = node_end[0]["payload"]
        assert isinstance(payload, dict)
        return float(payload["latency_ms"])

    serial_latency_ms = await run_with_concurrency(1)
    parallel_latency_ms = await run_with_concurrency(3)

    assert serial_latency_ms >= 140.0
    assert parallel_latency_ms < serial_latency_ms * 0.7


@pytest.mark.asyncio
async def test_researcher_isolates_one_failed_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_kb_search(query: str, top_n: int, *, trace_id: str) -> KBSearchResult:
        return make_kb_result(query)

    async def fake_summary(
        question: str,
        evidence: Sequence[dict[str, str]],
        system_prompt: str,
        trace_id: str,
    ) -> str:
        if question == "失败问题":
            raise RuntimeError("summary failed")
        return f"{question} 摘要"

    monkeypatch.setattr(researcher_module, "kb_search", fake_kb_search)
    monkeypatch.setattr(researcher_module, "_call_summary_model", fake_summary)

    result = await researcher_module.researcher_node(
        make_state(["成功问题", "失败问题", "另一个成功问题"])
    )

    assert set(result["research_results"]) == {"成功问题", "另一个成功问题"}
    assert len(result["errors"]) == 1
    assert "失败问题" in result["errors"][0]
    assert "summary failed" in result["errors"][0]
    assert "retry_count" not in result


@pytest.mark.asyncio
async def test_researcher_revision_only_researches_missing_aspects_and_counts_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    researched_targets: list[str] = []
    state = make_state(["原始问题 A", "原始问题 B"])
    state["research_results"] = {
        "原始问题 A": "原结果 A",
        "原始问题 B": "原结果 B",
    }
    state["citations"] = {
        "原始问题 A": [
            {"source": "kb/original.md", "origin": "kb", "snippet": "原证据"}
        ]
    }
    state["missing_aspects"] = ["补充 2026 行业数据"]
    state["quality_score"] = 0.4
    state["revision_count"] = 1

    def fake_kb_search(query: str, top_n: int, *, trace_id: str) -> KBSearchResult:
        researched_targets.append(query)
        return make_kb_result(query)

    async def fake_summary(
        question: str,
        evidence: Sequence[dict[str, str]],
        system_prompt: str,
        trace_id: str,
    ) -> str:
        return "定向补查结果"

    monkeypatch.setattr(researcher_module, "kb_search", fake_kb_search)
    monkeypatch.setattr(researcher_module, "_call_summary_model", fake_summary)

    result = await researcher_module.researcher_node(state)

    assert researched_targets == ["补充 2026 行业数据"]
    assert result["research_results"]["原始问题 A"] == "原结果 A"
    assert result["research_results"]["原始问题 B"] == "原结果 B"
    assert result["research_results"]["补充 2026 行业数据"].startswith("定向补查结果")
    assert result["citations"]["原始问题 A"] == state["citations"]["原始问题 A"]
    assert result["revision_count"] == 2
    assert result["missing_aspects"] == []
    assert "retry_count" not in result


@pytest.mark.asyncio
async def test_researcher_increments_technical_retry_only_when_all_initial_tasks_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_kb_search(query: str, top_n: int, *, trace_id: str) -> KBSearchResult:
        return {"hits": [], "max_score": 0.0}

    def failed_web_search(query: str, max_results: int) -> list[SearchResult]:
        raise RuntimeError("search unavailable")

    monkeypatch.setattr(researcher_module, "kb_search", failed_kb_search)
    monkeypatch.setattr(researcher_module, "web_search", failed_web_search)

    result = await researcher_module.researcher_node(make_state(["Q1", "Q2"]))

    assert result["research_results"] == {}
    assert result["retry_count"] == 1
    assert result["revision_count"] == 0
    assert len(result["errors"]) == 2


@pytest.mark.asyncio
async def test_researcher_writes_error_for_empty_targets() -> None:
    result = await researcher_module.researcher_node(make_state([]))

    assert result["errors"] == [
        "Researcher: sub_questions and missing_aspects must not both be empty"
    ]


@pytest.mark.asyncio
async def test_researcher_preserves_existing_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = make_state(["问题 A"])
    state["errors"] = ["Planner: previous error"]

    def failed_kb_search(query: str, top_n: int, *, trace_id: str) -> KBSearchResult:
        raise RuntimeError("kb unavailable")

    def failed_web_search(query: str, max_results: int) -> list[SearchResult]:
        raise RuntimeError("web unavailable")

    monkeypatch.setattr(researcher_module, "kb_search", failed_kb_search)
    monkeypatch.setattr(researcher_module, "web_search", failed_web_search)

    result = await researcher_module.researcher_node(state)

    assert result["errors"][0] == "Planner: previous error"
    assert result["errors"][1].startswith("Researcher: 问题 A")


@pytest.mark.asyncio
async def test_researcher_captures_prompt_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_prompt_load(name: str) -> str:
        raise RuntimeError("prompt unavailable")

    monkeypatch.setattr(researcher_module, "load_prompt", fail_prompt_load)

    result = await researcher_module.researcher_node(make_state(["问题 A"]))

    assert result["errors"] == ["Researcher: prompt unavailable"]
