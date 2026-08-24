from __future__ import annotations

import json

from agents.graph import build_graph
from agents.state import ResearchState
from backend.streaming import stream_research_progress
from frontend.app import apply_event_to_view_state, create_view_state, parse_sse_lines


def test_mock_research_flow_streams_into_frontend_state() -> None:
    def planner(state: ResearchState) -> dict[str, object]:
        assert state["topic"] == "AI Agent 岗位需求趋势"
        return {
            "sub_questions": [
                "AI Agent 岗位需求有哪些变化?",
                "企业需要哪些 AI Agent 技能?",
                "未来招聘趋势是什么?",
            ]
        }

    def researcher(state: ResearchState) -> dict[str, object]:
        return {
            "research_results": {
                question: f"{question} 的资料摘要。\n\n来源:\n- https://example.com"
                for question in state["sub_questions"]
            },
            "errors": [],
        }

    def writer(state: ResearchState) -> dict[str, object]:
        assert len(state["research_results"]) == 3
        return {
            "final_report": "\n".join(
                [
                    "# AI Agent 岗位需求趋势",
                    "",
                    "## 摘要",
                    "AI Agent 相关岗位需求正在增长。",
                    "",
                    "## 参考来源",
                    "- https://example.com",
                ]
            )
        }

    compiled_graph = build_graph(
        planner=planner,
        researcher=researcher,
        writer=writer,
    )

    raw_sse_events = list(
        stream_research_progress(
            "AI Agent 岗位需求趋势",
            compiled_graph=compiled_graph,
        )
    )
    frontend_events = list(parse_sse_lines(_to_sse_lines(raw_sse_events)))
    view_state = create_view_state()
    for event in frontend_events:
        apply_event_to_view_state(event, view_state)

    assert [event["event"] for event in frontend_events] == [
        "start",
        "progress",
        "progress",
        "progress",
        "complete",
    ]
    assert [event["data"]["node"] for event in frontend_events[1:4]] == [
        "planner",
        "researcher",
        "writer",
    ]
    assert view_state["agent_status"]["planner"]["status"] == "完成"
    assert view_state["agent_status"]["researcher"]["status"] == "完成"
    assert view_state["agent_status"]["writer"]["status"] == "完成"
    assert len(view_state["sub_questions"]) == 3
    assert view_state["research_result_count"] == 3
    assert "AI Agent 相关岗位需求正在增长" in view_state["final_report"]
    assert view_state["errors"] == []


def _to_sse_lines(events: list[dict[str, str]]) -> list[str]:
    lines: list[str] = []
    for event in events:
        json.loads(event["data"])
        lines.extend(
            [
                f"event: {event['event']}",
                f"data: {event['data']}",
                "",
            ]
        )
    return lines
