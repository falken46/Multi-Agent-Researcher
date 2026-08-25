from __future__ import annotations

import pytest

from agents.graph import build_graph, create_initial_state, should_retry
from agents.state import ResearchState
from core.config import clear_settings_cache


def test_create_initial_state() -> None:
    state = create_initial_state("测试主题", trace_id="trace-fixed")

    assert state == {
        "topic": "测试主题",
        "sub_questions": [],
        "research_results": {},
        "final_report": "",
        "errors": [],
        "retry_count": 0,
        "citations": {},
        "critique": "",
        "quality_score": 0.0,
        "quality_history": [],
        "missing_aspects": [],
        "revision_count": 0,
        "trace_id": "trace-fixed",
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "llm_calls": 0,
            "total_latency_ms": 0.0,
        },
        "fallback_queries": [],
    }


@pytest.mark.asyncio
async def test_graph_happy_path_passes_through_critic() -> None:
    calls: list[str] = []

    def fake_planner(state: ResearchState) -> dict[str, object]:
        calls.append("planner")
        return {"sub_questions": ["Q1", "Q2", "Q3"]}

    async def fake_researcher(state: ResearchState) -> dict[str, object]:
        calls.append("researcher")
        return {
            "research_results": {"Q1": "R1", "Q2": "R2", "Q3": "R3"},
            "errors": [],
        }

    async def fake_critic(state: ResearchState) -> dict[str, object]:
        calls.append("critic")
        assert state["research_results"]["Q1"] == "R1"
        return {
            "quality_score": 0.9,
            "quality_history": [0.9],
            "critique": "资料完整",
            "missing_aspects": [],
        }

    def fake_writer(state: ResearchState) -> dict[str, object]:
        calls.append("writer")
        assert state["quality_score"] == 0.9
        return {"final_report": "# 测试报告"}

    test_graph = build_graph(
        planner=fake_planner,
        researcher=fake_researcher,
        critic=fake_critic,
        writer=fake_writer,
    )

    final_state = await test_graph.ainvoke(create_initial_state("测试主题"))

    assert calls == ["planner", "researcher", "critic", "writer"]
    assert final_state["final_report"] == "# 测试报告"
    assert final_state["revision_count"] == 0


@pytest.mark.asyncio
async def test_graph_retries_researcher_until_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_RETRY", "2")
    clear_settings_cache()
    researcher_calls = 0

    def fake_planner(state: ResearchState) -> dict[str, object]:
        return {"sub_questions": ["Q1", "Q2", "Q3"]}

    async def fake_researcher(state: ResearchState) -> dict[str, object]:
        nonlocal researcher_calls
        researcher_calls += 1
        if researcher_calls == 1:
            return {
                "research_results": {},
                "errors": ["Researcher: transient failure"],
                "retry_count": state["retry_count"] + 1,
            }
        return {"research_results": {"Q1": "R1"}, "errors": state["errors"]}

    async def fake_critic(state: ResearchState) -> dict[str, object]:
        return {
            "quality_score": 0.9,
            "quality_history": [0.9],
            "critique": "足够",
            "missing_aspects": [],
        }

    def fake_writer(state: ResearchState) -> dict[str, object]:
        return {"final_report": "# 重试后报告"}

    test_graph = build_graph(
        planner=fake_planner,
        researcher=fake_researcher,
        critic=fake_critic,
        writer=fake_writer,
    )

    final_state = await test_graph.ainvoke(create_initial_state("测试主题"))

    assert researcher_calls == 2
    assert final_state["retry_count"] == 1
    assert final_state["revision_count"] == 0
    assert final_state["final_report"] == "# 重试后报告"


@pytest.mark.asyncio
async def test_graph_stops_technical_retry_after_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_RETRY", "2")
    clear_settings_cache()
    researcher_calls = 0
    critic_calls = 0

    def fake_planner(state: ResearchState) -> dict[str, object]:
        return {"sub_questions": ["Q1", "Q2", "Q3"]}

    async def fake_researcher(state: ResearchState) -> dict[str, object]:
        nonlocal researcher_calls
        researcher_calls += 1
        return {
            "research_results": {},
            "errors": [*state["errors"], f"Researcher: failure {researcher_calls}"],
            "retry_count": state["retry_count"] + 1,
        }

    async def unexpected_critic(state: ResearchState) -> dict[str, object]:
        nonlocal critic_calls
        critic_calls += 1
        raise AssertionError("无研究结果时不应调用 Critic")

    def fake_writer(state: ResearchState) -> dict[str, object]:
        return {"final_report": "# 带错误报告"}

    test_graph = build_graph(
        planner=fake_planner,
        researcher=fake_researcher,
        critic=unexpected_critic,
        writer=fake_writer,
    )

    final_state = await test_graph.ainvoke(create_initial_state("测试主题"))

    assert researcher_calls == 2
    assert critic_calls == 0
    assert final_state["retry_count"] == 2
    assert final_state["revision_count"] == 0
    assert len(final_state["errors"]) == 2
    assert final_state["final_report"] == "# 带错误报告"


def test_should_retry_routes_to_continue_when_results_exist() -> None:
    state = create_initial_state("测试主题")
    state["sub_questions"] = ["Q1"]
    state["research_results"] = {"Q1": "R1"}

    assert should_retry(state) == "continue"
