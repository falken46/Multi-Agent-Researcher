from __future__ import annotations

from agents.graph import build_graph, create_initial_state, should_retry
from agents.state import ResearchState


def test_create_initial_state() -> None:
    state = create_initial_state("测试主题")

    assert state == {
        "topic": "测试主题",
        "sub_questions": [],
        "research_results": {},
        "final_report": "",
        "errors": [],
        "retry_count": 0,
    }


def test_graph_happy_path() -> None:
    calls: list[str] = []

    def fake_planner(state: ResearchState) -> dict[str, object]:
        calls.append("planner")
        return {"sub_questions": ["Q1", "Q2", "Q3"]}

    def fake_researcher(state: ResearchState) -> dict[str, object]:
        calls.append("researcher")
        return {
            "research_results": {
                "Q1": "R1",
                "Q2": "R2",
                "Q3": "R3",
            },
            "errors": [],
        }

    def fake_writer(state: ResearchState) -> dict[str, object]:
        calls.append("writer")
        assert state["research_results"]["Q1"] == "R1"
        return {"final_report": "# 测试报告"}

    test_graph = build_graph(
        planner=fake_planner,
        researcher=fake_researcher,
        writer=fake_writer,
    )

    final_state = test_graph.invoke(create_initial_state("测试主题"))

    assert calls == ["planner", "researcher", "writer"]
    assert final_state["final_report"] == "# 测试报告"


def test_graph_retries_researcher_until_results(monkeypatch) -> None:
    monkeypatch.setenv("MAX_RETRY", "2")
    researcher_calls = 0

    def fake_planner(state: ResearchState) -> dict[str, object]:
        return {"sub_questions": ["Q1", "Q2", "Q3"]}

    def fake_researcher(state: ResearchState) -> dict[str, object]:
        nonlocal researcher_calls
        researcher_calls += 1
        if researcher_calls == 1:
            return {
                "research_results": {},
                "errors": ["Researcher: transient failure"],
                "retry_count": state["retry_count"] + 1,
            }
        return {
            "research_results": {"Q1": "R1"},
            "errors": state["errors"],
        }

    def fake_writer(state: ResearchState) -> dict[str, object]:
        return {"final_report": "# 重试后报告"}

    test_graph = build_graph(
        planner=fake_planner,
        researcher=fake_researcher,
        writer=fake_writer,
    )

    final_state = test_graph.invoke(create_initial_state("测试主题"))

    assert researcher_calls == 2
    assert final_state["retry_count"] == 1
    assert final_state["final_report"] == "# 重试后报告"


def test_graph_stops_retry_after_limit(monkeypatch) -> None:
    monkeypatch.setenv("MAX_RETRY", "2")
    researcher_calls = 0

    def fake_planner(state: ResearchState) -> dict[str, object]:
        return {"sub_questions": ["Q1", "Q2", "Q3"]}

    def fake_researcher(state: ResearchState) -> dict[str, object]:
        nonlocal researcher_calls
        researcher_calls += 1
        return {
            "research_results": {},
            "errors": [*state["errors"], f"Researcher: failure {researcher_calls}"],
            "retry_count": state["retry_count"] + 1,
        }

    def fake_writer(state: ResearchState) -> dict[str, object]:
        return {"final_report": "# 带错误报告"}

    test_graph = build_graph(
        planner=fake_planner,
        researcher=fake_researcher,
        writer=fake_writer,
    )

    final_state = test_graph.invoke(create_initial_state("测试主题"))

    assert researcher_calls == 2
    assert final_state["retry_count"] == 2
    assert len(final_state["errors"]) == 2
    assert final_state["final_report"] == "# 带错误报告"


def test_should_retry_routes_to_continue_when_results_exist() -> None:
    state = create_initial_state("测试主题")
    state["sub_questions"] = ["Q1"]
    state["research_results"] = {"Q1": "R1"}

    assert should_retry(state) == "continue"
