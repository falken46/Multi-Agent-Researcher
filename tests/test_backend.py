from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import backend.api as api_module
import backend.streaming as streaming_module
from backend.api import ResearchRequest, create_app
from backend.streaming import stream_research_progress


def _payload(event: dict[str, str]) -> dict[str, Any]:
    return json.loads(event["data"])


def test_health_endpoint() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_research_request_normalizes_resume_fields() -> None:
    request = ResearchRequest(
        topic="  test topic  ",
        thread_id="checkpoint-thread",
        resume=True,
    )

    assert request.topic == "test topic"
    assert request.thread_id == "checkpoint-thread"
    assert request.resume is True


def test_research_request_requires_thread_id_when_resuming() -> None:
    with pytest.raises(ValidationError, match="thread_id"):
        ResearchRequest(topic="test topic", resume=True)


def test_research_endpoint_streams_sse_and_forwards_resume_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str | None, bool]] = []

    async def fake_stream(
        topic: str,
        *,
        thread_id: str | None = None,
        resume: bool = False,
    ):
        calls.append((topic, thread_id, resume))
        yield {"event": "start", "data": json.dumps({"node": "start"})}
        yield {"event": "complete", "data": json.dumps({"node": "end"})}

    monkeypatch.setattr(api_module, "stream_research_progress", fake_stream)
    client = TestClient(create_app())

    with client.stream(
        "POST",
        "/research",
        json={
            "topic": " test topic ",
            "thread_id": "checkpoint-thread",
            "resume": True,
        },
    ) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: start" in body
    assert "event: complete" in body
    assert calls == [("test topic", "checkpoint-thread", True)]


def test_research_endpoint_rejects_blank_topic() -> None:
    client = TestClient(create_app())

    response = client.post("/research", json={"topic": "   "})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_stream_research_progress_emits_v2_updates_custom_and_trace_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 25,
        "total_tokens": 125,
        "total_cost": 0.0015,
        "llm_calls": 2,
        "total_latency_ms": 321.5,
    }
    trace_summary = {
        "trace_id": "thread-123",
        "event_count": 12,
        **expected_usage,
        "fallback_count": 1,
        "revision_count": 1,
        "errors": [],
    }
    summarized_trace_ids: list[str] = []

    def fake_summarize(trace_id: str) -> dict[str, Any]:
        summarized_trace_ids.append(trace_id)
        return trace_summary

    monkeypatch.setattr(streaming_module, "summarize", fake_summarize)

    class FakeGraph:
        async def astream(
            self,
            graph_input: dict[str, Any],
            config: dict[str, Any],
            **kwargs: Any,
        ):
            assert graph_input["topic"] == "test topic"
            assert graph_input["trace_id"] == "thread-123"
            assert config["configurable"]["thread_id"] == "thread-123"
            assert kwargs["stream_mode"] == ["updates", "custom"]
            assert kwargs["version"] == "v2"
            assert kwargs["durability"] == "sync"
            yield {
                "type": "updates",
                "ns": [],
                "data": {"planner": {"sub_questions": ["Q1"]}},
            }
            yield {
                "type": "custom",
                "ns": [],
                "data": {
                    "event": "fallback",
                    "payload": {"query": "Q1", "reason": "low_score"},
                },
            }
            yield {
                "type": "updates",
                "ns": [],
                "data": {
                    "researcher": {
                        "research_results": {"Q1": "R1"},
                        "errors": [],
                        "fallback_queries": ["Q1"],
                    }
                },
            }
            yield {
                "type": "custom",
                "ns": [],
                "data": {
                    "event": "critic_start",
                    "payload": {"revision_count": 0},
                },
            }
            yield {
                "type": "custom",
                "ns": [],
                "data": {
                    "event": "critic_done",
                    "payload": {
                        "quality_score": 0.4,
                        "critique": "coverage is incomplete",
                        "missing_aspects": ["Q2"],
                    },
                },
            }
            yield {
                "type": "updates",
                "ns": [],
                "data": {
                    "critic": {
                        "quality_score": 0.4,
                        "quality_history": [0.4],
                        "critique": "coverage is incomplete",
                        "missing_aspects": ["Q2"],
                    }
                },
            }
            yield {
                "type": "custom",
                "ns": [],
                "data": {
                    "event": "revision",
                    "payload": {"revision_count": 1, "missing_aspects": ["Q2"]},
                },
            }
            yield {
                "type": "updates",
                "ns": [],
                "data": {
                    "writer": {
                        "quality_score": 0.8,
                        "missing_aspects": [],
                        "revision_count": 1,
                        "final_report": "# report",
                    }
                },
            }

    events = [
        event
        async for event in stream_research_progress(
            " test topic ",
            compiled_graph=FakeGraph(),
            thread_id="thread-123",
        )
    ]

    event_names = [event["event"] for event in events]
    assert event_names == [
        "start",
        "progress",
        "fallback",
        "progress",
        "critic_start",
        "critic_done",
        "progress",
        "revision",
        "progress",
        "usage",
        "complete",
    ]
    assert summarized_trace_ids == ["thread-123"]
    usage_payload = _payload(events[-2])
    assert usage_payload["usage"] == expected_usage
    complete_payload = _payload(events[-1])
    assert complete_payload["state"]["final_report"] == "# report"
    assert complete_payload["state"]["research_result_count"] == 1
    assert complete_payload["state"]["usage"] == expected_usage
    assert complete_payload["trace"] == trace_summary


@pytest.mark.asyncio
async def test_stream_research_progress_resumes_same_thread_with_none_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persisted_state = {
        "topic": "persisted topic",
        "sub_questions": ["Q1"],
        "research_results": {"Q1": "R1"},
        "final_report": "",
        "errors": [],
        "retry_count": 0,
        "citations": {},
        "critique": "",
        "quality_score": 0.0,
        "quality_history": [],
        "missing_aspects": [],
        "revision_count": 0,
        "trace_id": "resume-thread",
        "usage": {},
        "fallback_queries": [],
    }

    monkeypatch.setattr(
        streaming_module,
        "summarize",
        lambda trace_id: {
            "trace_id": trace_id,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "llm_calls": 0,
            "total_latency_ms": 0.0,
        },
    )

    class ResumeGraph:
        async def aget_state(self, config: dict[str, Any]):
            assert config["configurable"]["thread_id"] == "resume-thread"
            return SimpleNamespace(values=persisted_state)

        async def astream(
            self,
            graph_input: None,
            config: dict[str, Any],
            **kwargs: Any,
        ):
            assert graph_input is None
            assert config["configurable"]["thread_id"] == "resume-thread"
            assert kwargs["version"] == "v2"
            assert kwargs["durability"] == "sync"
            yield {
                "type": "updates",
                "ns": [],
                "data": {"writer": {"final_report": "# resumed report"}},
            }

    events = [
        event
        async for event in stream_research_progress(
            "persisted topic",
            compiled_graph=ResumeGraph(),
            thread_id="resume-thread",
            resume=True,
        )
    ]

    start_payload = _payload(events[0])
    complete_payload = _payload(events[-1])
    assert start_payload["resumed"] is True
    assert start_payload["thread_id"] == "resume-thread"
    assert complete_payload["state"]["topic"] == "persisted topic"
    assert complete_payload["state"]["final_report"] == "# resumed report"


@pytest.mark.asyncio
async def test_stream_research_progress_marks_failed_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        streaming_module,
        "summarize",
        lambda trace_id: {
            "trace_id": trace_id,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "llm_calls": 0,
            "total_latency_ms": 0.0,
        },
    )

    class FakeGraph:
        async def astream(
            self,
            graph_input: dict[str, Any],
            config: dict[str, Any],
            **kwargs: Any,
        ):
            yield {
                "type": "updates",
                "ns": [],
                "data": {
                    "planner": {"errors": ["Planner: connection error"]},
                },
            }
            yield {
                "type": "updates",
                "ns": [],
                "data": {
                    "writer": {
                        "errors": [
                            "Planner: connection error",
                            "Writer: sub_questions must not be empty",
                        ]
                    }
                },
            }

    events = [
        event
        async for event in stream_research_progress(
            "test topic",
            compiled_graph=FakeGraph(),
        )
    ]
    progress_payloads = [
        _payload(event) for event in events if event["event"] == "progress"
    ]
    complete_payload = _payload(events[-1])

    assert progress_payloads[0]["status"] == "failed"
    assert progress_payloads[1]["status"] == "failed"
    assert complete_payload["status"] == "failed"
    assert complete_payload["state"]["final_report"] == ""
