from __future__ import annotations

import json

from fastapi.testclient import TestClient

import backend.api as api_module
from backend.api import create_app
from backend.streaming import stream_research_progress


def test_health_endpoint() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_research_endpoint_streams_sse(monkeypatch) -> None:
    def fake_stream(topic: str):
        assert topic == "测试主题"
        yield {"event": "start", "data": json.dumps({"node": "start"})}
        yield {"event": "complete", "data": json.dumps({"node": "end"})}

    monkeypatch.setattr(api_module, "stream_research_progress", fake_stream)
    client = TestClient(create_app())

    with client.stream("POST", "/research", json={"topic": " 测试主题 "}) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: start" in body
    assert "event: complete" in body


def test_research_endpoint_rejects_blank_topic() -> None:
    client = TestClient(create_app())

    response = client.post("/research", json={"topic": "   "})

    assert response.status_code == 422


def test_stream_research_progress_emits_graph_updates() -> None:
    class FakeGraph:
        def stream(self, state, stream_mode: str):
            assert stream_mode == "updates"
            assert state["topic"] == "测试主题"
            yield {"planner": {"sub_questions": ["Q1", "Q2", "Q3"]}}
            yield {"researcher": {"research_results": {"Q1": "R1"}, "errors": []}}
            yield {"writer": {"final_report": "# 报告"}}

    events = list(stream_research_progress("测试主题", compiled_graph=FakeGraph()))

    assert [event["event"] for event in events] == [
        "start",
        "progress",
        "progress",
        "progress",
        "complete",
    ]
    complete_payload = json.loads(events[-1]["data"])
    assert complete_payload["state"]["final_report"] == "# 报告"
    assert complete_payload["state"]["research_result_count"] == 1
