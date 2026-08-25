from __future__ import annotations

from collections.abc import Iterator

import pytest

from agents.graph import build_graph, create_initial_state
from agents.state import ResearchState
from core.config import clear_settings_cache


def _planner(state: ResearchState) -> dict[str, object]:
    return {"sub_questions": ["Q1"]}


def _writer(state: ResearchState) -> dict[str, object]:
    return {"final_report": "# 完成"}


def _critic_result(
    state: ResearchState,
    *,
    score: float,
    missing: list[str],
) -> dict[str, object]:
    return {
        "quality_score": score,
        "quality_history": [*state["quality_history"], score],
        "critique": f"score={score}",
        "missing_aspects": missing,
    }


def _next_score(scores: Iterator[float]) -> float:
    try:
        return next(scores)
    except StopIteration as exc:
        raise AssertionError("Critic 被调用次数超过预期") from exc


@pytest.mark.asyncio
async def test_low_score_directed_revision_then_higher_score_reaches_writer() -> None:
    calls: list[str] = []
    critic_scores = iter([0.4, 0.85])

    def planner(state: ResearchState) -> dict[str, object]:
        calls.append("planner")
        return _planner(state)

    async def researcher(state: ResearchState) -> dict[str, object]:
        calls.append("researcher")
        if state["missing_aspects"]:
            assert state["missing_aspects"] == ["补充工程落地风险"]
            return {
                "research_results": {
                    **state["research_results"],
                    "补充工程落地风险": "定向补查资料",
                },
                "revision_count": state["revision_count"] + 1,
                "missing_aspects": [],
            }
        return {"research_results": {"Q1": "初始资料"}}

    async def critic(state: ResearchState) -> dict[str, object]:
        calls.append("critic")
        score = _next_score(critic_scores)
        missing = ["补充工程落地风险"] if score < 0.7 else []
        return _critic_result(state, score=score, missing=missing)

    def writer(state: ResearchState) -> dict[str, object]:
        calls.append("writer")
        assert state["research_results"]["补充工程落地风险"] == "定向补查资料"
        return _writer(state)

    graph = build_graph(
        planner=planner,
        researcher=researcher,
        critic=critic,
        writer=writer,
    )

    final_state = await graph.ainvoke(create_initial_state("测试主题"))

    assert calls == [
        "planner",
        "researcher",
        "critic",
        "researcher",
        "critic",
        "writer",
    ]
    assert final_state["quality_history"] == [0.4, 0.85]
    assert final_state["quality_score"] == pytest.approx(0.85)
    assert final_state["revision_count"] == 1
    assert final_state["final_report"] == "# 完成"


@pytest.mark.asyncio
async def test_always_low_score_is_capped_by_max_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_REVISION", "2")
    clear_settings_cache()
    researcher_calls = 0
    critic_calls = 0

    async def researcher(state: ResearchState) -> dict[str, object]:
        nonlocal researcher_calls
        researcher_calls += 1
        if state["missing_aspects"]:
            round_number = state["revision_count"] + 1
            return {
                "research_results": {
                    **state["research_results"],
                    f"缺口-{round_number}": f"补查-{round_number}",
                },
                "revision_count": round_number,
                "missing_aspects": [],
            }
        return {"research_results": {"Q1": "初始资料"}}

    async def critic(state: ResearchState) -> dict[str, object]:
        nonlocal critic_calls
        critic_calls += 1
        return _critic_result(
            state,
            score=0.2 + critic_calls * 0.01,
            missing=[f"缺口-{critic_calls}"],
        )

    graph = build_graph(
        planner=_planner,
        researcher=researcher,
        critic=critic,
        writer=_writer,
    )

    final_state = await graph.ainvoke(create_initial_state("测试主题"))

    assert researcher_calls == 3  # 初查一次 + 最多返工两次
    assert critic_calls == 3
    assert final_state["revision_count"] == 2
    assert final_state["quality_score"] < 0.7
    assert final_state["final_report"] == "# 完成"


@pytest.mark.asyncio
async def test_quality_stall_exits_before_hard_revision_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_REVISION", "5")
    clear_settings_cache()
    scores = iter([0.3, 0.4, 0.4])
    researcher_calls = 0
    critic_calls = 0

    async def researcher(state: ResearchState) -> dict[str, object]:
        nonlocal researcher_calls
        researcher_calls += 1
        if state["missing_aspects"]:
            return {
                "research_results": {
                    **state["research_results"],
                    f"补查-{researcher_calls}": "没有带来进一步提升",
                },
                "revision_count": state["revision_count"] + 1,
                "missing_aspects": [],
            }
        return {"research_results": {"Q1": "初始资料"}}

    async def critic(state: ResearchState) -> dict[str, object]:
        nonlocal critic_calls
        critic_calls += 1
        score = _next_score(scores)
        return _critic_result(state, score=score, missing=["仍有缺口"])

    graph = build_graph(
        planner=_planner,
        researcher=researcher,
        critic=critic,
        writer=_writer,
    )

    final_state = await graph.ainvoke(create_initial_state("测试主题"))

    assert researcher_calls == 3
    assert critic_calls == 3
    assert final_state["revision_count"] == 2
    assert final_state["revision_count"] < 5
    assert final_state["quality_history"] == [0.3, 0.4, 0.4]
    assert final_state["final_report"] == "# 完成"


@pytest.mark.asyncio
async def test_technical_retry_and_quality_revision_use_separate_counters() -> None:
    researcher_calls = 0
    critic_calls = 0

    async def researcher(state: ResearchState) -> dict[str, object]:
        nonlocal researcher_calls
        researcher_calls += 1
        if researcher_calls == 1:
            return {
                "research_results": {},
                "retry_count": state["retry_count"] + 1,
                "errors": ["Researcher: temporary failure"],
            }
        if state["missing_aspects"]:
            return {
                "research_results": {
                    **state["research_results"],
                    "补充风险": "补查结果",
                },
                "revision_count": state["revision_count"] + 1,
                "missing_aspects": [],
            }
        return {"research_results": {"Q1": "重试后的初始资料"}}

    async def critic(state: ResearchState) -> dict[str, object]:
        nonlocal critic_calls
        critic_calls += 1
        score = 0.4 if critic_calls == 1 else 0.8
        return _critic_result(
            state,
            score=score,
            missing=["补充风险"] if score < 0.7 else [],
        )

    graph = build_graph(
        planner=_planner,
        researcher=researcher,
        critic=critic,
        writer=_writer,
    )

    final_state = await graph.ainvoke(create_initial_state("测试主题"))

    assert researcher_calls == 3
    assert critic_calls == 2
    assert final_state["retry_count"] == 1
    assert final_state["revision_count"] == 1
    assert final_state["quality_history"] == [0.4, 0.8]
