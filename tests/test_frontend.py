from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

frontend_app = importlib.import_module("frontend.app")


def test_parse_sse_lines_parses_json_and_ignores_comments() -> None:
    lines = [
        "event: start",
        'data: {"node": "start", "topic": "AI Agent"}',
        "",
        ": ping",
        "",
        "event: progress",
        'data: {"node": "planner", "state": {"sub_questions": ["Q1"], '
        '"research_result_count": 0, "final_report": "", "errors": [], "retry_count": 0}}',
        "",
    ]

    events = list(frontend_app.parse_sse_lines(lines))

    assert [event["event"] for event in events] == ["start", "progress"]
    assert events[0]["data"]["topic"] == "AI Agent"
    assert events[1]["data"]["state"]["sub_questions"] == ["Q1"]


def test_parse_sse_lines_keeps_plain_text_data() -> None:
    lines = ["event: error", "data: backend unavailable", ""]

    events = list(frontend_app.parse_sse_lines(lines))

    assert events == [{"event": "error", "data": "backend unavailable"}]


def test_apply_event_to_view_state_tracks_agent_progress() -> None:
    view_state = frontend_app.create_view_state("AI Agent")

    frontend_app.apply_event_to_view_state(
        {"event": "start", "data": {"node": "start", "topic": "AI Agent"}},
        view_state,
    )
    frontend_app.apply_event_to_view_state(
        {
            "event": "progress",
            "data": {
                "node": "planner",
                "state": {
                    "topic": "AI Agent",
                    "sub_questions": ["Q1", "Q2", "Q3"],
                    "research_result_count": 0,
                    "final_report": "",
                    "errors": [],
                    "retry_count": 0,
                },
            },
        },
        view_state,
    )
    frontend_app.apply_event_to_view_state(
        {
            "event": "complete",
            "data": {
                "node": "end",
                "state": {
                    "topic": "AI Agent",
                    "sub_questions": ["Q1", "Q2", "Q3"],
                    "research_result_count": 3,
                    "final_report": "# 报告",
                    "errors": [],
                    "retry_count": 0,
                },
            },
        },
        view_state,
    )

    assert view_state["agent_status"]["planner"]["status"] == "完成"
    assert view_state["agent_status"]["researcher"]["status"] == "完成"
    assert view_state["agent_status"]["critic"]["status"] == "完成"
    assert view_state["agent_status"]["writer"]["status"] == "完成"
    assert view_state["research_result_count"] == 3
    assert view_state["final_report"] == "# 报告"


def test_apply_event_to_view_state_records_error() -> None:
    view_state = frontend_app.create_view_state("AI Agent")

    frontend_app.apply_event_to_view_state(
        {
            "event": "error",
            "data": {"node": "researcher", "error": "search failed"},
        },
        view_state,
    )

    assert view_state["agent_status"]["researcher"]["status"] == "失败"
    assert view_state["errors"] == ["search failed"]


def test_apply_event_to_view_state_does_not_mark_empty_failure_complete() -> None:
    view_state = frontend_app.create_view_state("AI Agent")

    frontend_app.apply_event_to_view_state(
        {
            "event": "progress",
            "data": {
                "node": "planner",
                "status": "failed",
                "state": {
                    "topic": "AI Agent",
                    "sub_questions": [],
                    "research_result_count": 0,
                    "final_report": "",
                    "errors": ["Planner: Connection error."],
                    "retry_count": 0,
                },
            },
        },
        view_state,
    )
    frontend_app.apply_event_to_view_state(
        {
            "event": "complete",
            "data": {
                "node": "end",
                "status": "failed",
                "state": {
                    "topic": "AI Agent",
                    "sub_questions": [],
                    "research_result_count": 0,
                    "final_report": "",
                    "errors": ["Planner: Connection error."],
                    "retry_count": 0,
                },
            },
        },
        view_state,
    )

    assert view_state["agent_status"]["planner"]["status"] == "失败"
    assert view_state["agent_status"]["researcher"]["status"] == "阻塞"
    assert view_state["agent_status"]["critic"]["status"] == "阻塞"
    assert view_state["agent_status"]["writer"]["status"] == "阻塞"
    assert frontend_app._overall_progress(view_state) == 0


def test_apply_event_tracks_critic_revision_fallback_and_usage() -> None:
    view_state = frontend_app.create_view_state("AI Agent")

    frontend_app.apply_event_to_view_state(
        {
            "event": "critic_start",
            "data": {"node": "critic", "research_result_count": 3},
        },
        view_state,
    )
    assert view_state["agent_status"]["critic"]["status"] == "运行中"

    frontend_app.apply_event_to_view_state(
        {
            "event": "critic_done",
            "data": {
                "node": "critic",
                "quality_score": 0.4,
                "critique": "缺少工程实践证据",
                "missing_aspects": ["补充并发测试"],
            },
        },
        view_state,
    )

    assert view_state["quality_score"] == pytest.approx(0.4)
    assert view_state["critique"] == "缺少工程实践证据"
    assert view_state["missing_aspects"] == ["补充并发测试"]
    # Critic 的条件边尚未产生真实事件，不能抢跑显示 Writer 正在运行。
    assert view_state["agent_status"]["writer"]["status"] == "等待"

    frontend_app.apply_event_to_view_state(
        {
            "event": "revision",
            "data": {
                "node": "researcher",
                "round": 1,
                "missing_aspects": ["补充并发测试"],
            },
        },
        view_state,
    )
    frontend_app.apply_event_to_view_state(
        {
            "event": "fallback",
            "data": {
                "node": "researcher",
                "query": "补充并发测试",
                "reason": "low_score",
            },
        },
        view_state,
    )
    frontend_app.apply_event_to_view_state(
        {
            "event": "usage",
            "data": {
                "usage": {
                    "total_tokens": 321,
                    "total_cost": 0.0123,
                    "total_latency_ms": 2500,
                    "llm_calls": 4,
                },
                "fallback_count": 1,
                "revision_count": 1,
            },
        },
        view_state,
    )

    assert view_state["revision_count"] == 1
    assert view_state["fallback_count"] == 1
    assert view_state["fallback_queries"] == ["补充并发测试"]
    assert view_state["usage"]["total_tokens"] == 321
    assert view_state["agent_status"]["researcher"]["status"] == "运行中"


def test_subscribe_research_posts_to_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self, decode_unicode: bool):
            assert decode_unicode is True
            return iter(
                [
                    "event: start",
                    'data: {"node": "start"}',
                    "",
                ]
            )

    def fake_post(url, json, headers, stream, timeout):
        assert url == "http://backend/research"
        assert json == {"topic": "AI Agent"}
        assert headers == {"Accept": "text/event-stream"}
        assert stream is True
        assert timeout == frontend_app.REQUEST_TIMEOUT
        return FakeResponse()

    monkeypatch.setattr(frontend_app.requests, "post", fake_post)

    events = list(frontend_app.subscribe_research("AI Agent", api_base_url="http://backend"))

    assert events == [{"event": "start", "data": {"node": "start"}}]


def test_streamlit_app_runs_research_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self, decode_unicode: bool):
            assert decode_unicode is True
            return iter(
                [
                    "event: start",
                    'data: {"node": "start", "status": "started", "topic": "AI Agent"}',
                    "",
                    "event: progress",
                    'data: {"node": "planner", "status": "completed", '
                    '"state": {"topic": "AI Agent", "sub_questions": ["Q1", "Q2", "Q3"], '
                    '"research_result_count": 0, "final_report": "", "errors": [], '
                    '"retry_count": 0}}',
                    "",
                    "event: progress",
                    'data: {"node": "researcher", "status": "completed", '
                    '"state": {"topic": "AI Agent", "sub_questions": ["Q1", "Q2", "Q3"], '
                    '"research_result_count": 3, "final_report": "", "errors": [], '
                    '"retry_count": 0}}',
                    "",
                    "event: critic_start",
                    'data: {"node": "critic", "research_result_count": 3}',
                    "",
                    "event: critic_done",
                    'data: {"node": "critic", "quality_score": 0.9, '
                    '"critique": "资料完整", "missing_aspects": []}',
                    "",
                    "event: progress",
                    'data: {"node": "critic", "status": "completed", '
                    '"state": {"topic": "AI Agent", "sub_questions": ["Q1", "Q2", "Q3"], '
                    '"research_result_count": 3, "final_report": "", "errors": [], '
                    '"retry_count": 0, "quality_score": 0.9, "critique": "资料完整", '
                    '"missing_aspects": [], "revision_count": 0}}',
                    "",
                    "event: progress",
                    'data: {"node": "writer", "status": "completed", '
                    '"state": {"topic": "AI Agent", "sub_questions": ["Q1", "Q2", "Q3"], '
                    '"research_result_count": 3, "final_report": "# AI Agent\\n\\n## 摘要\\n测试报告", '
                    '"errors": [], "retry_count": 0}}',
                    "",
                    "event: usage",
                    'data: {"total_tokens": 321, "total_cost": 0.0123, '
                    '"total_latency_ms": 2500, "llm_calls": 4}',
                    "",
                    "event: complete",
                    'data: {"node": "end", "status": "completed", '
                    '"state": {"topic": "AI Agent", "sub_questions": ["Q1", "Q2", "Q3"], '
                    '"research_result_count": 3, "final_report": "# AI Agent\\n\\n## 摘要\\n测试报告", '
                    '"errors": [], "retry_count": 0}}',
                    "",
                ]
            )

    def fake_post(url, json, headers, stream, timeout):
        assert url == "http://127.0.0.1:8000/research"
        assert json == {"topic": "AI Agent"}
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    app_path = Path(__file__).parents[1] / "frontend" / "app.py"

    app = AppTest.from_file(str(app_path))
    app.run(timeout=5)
    app.text_input[0].set_value("AI Agent")
    app.button[0].click()
    app.run(timeout=10)

    rendered_reports = [
        markdown.value for markdown in app.markdown if "测试报告" in markdown.value
    ]
    assert rendered_reports == ["# AI Agent\n\n## 摘要\n测试报告"]


def test_report_filename_sanitizes_topic() -> None:
    assert frontend_app._report_filename("AI Agent 趋势 / 2026") == "AI_Agent_趋势_2026.md"
