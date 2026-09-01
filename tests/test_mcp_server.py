from __future__ import annotations

import json
from typing import Any

import pytest
from mcp import Client

import mcp_server.server as server_module


@pytest.mark.asyncio
async def test_mcp_lists_two_tools_with_llm_facing_schemas() -> None:
    async with Client(server_module.server) as client:
        response = await client.list_tools(cache_mode="reload")

    tools = {tool.name: tool for tool in response.tools}
    assert set(tools) == {"deep_research", "kb_search"}

    deep_research = tools["deep_research"]
    deep_properties = deep_research.input_schema["properties"]
    assert deep_research.input_schema["required"] == ["topic"]
    assert deep_properties["topic"]["minLength"] == 1
    assert deep_properties["topic"]["maxLength"] == 500
    assert "Markdown" in deep_properties["topic"]["description"]
    assert deep_properties["thread_id"]["anyOf"][0]["pattern"].startswith("^")
    assert "final_report" in deep_research.output_schema["properties"]
    assert deep_research.annotations is not None
    assert deep_research.annotations.read_only_hint is False
    assert deep_research.annotations.open_world_hint is True

    kb_search = tools["kb_search"]
    kb_properties = kb_search.input_schema["properties"]
    assert kb_search.input_schema["required"] == ["query"]
    assert kb_properties["top_n"]["minimum"] == 1
    assert kb_properties["top_n"]["maximum"] == 20
    assert "fallback_confidence" in kb_search.output_schema["properties"]
    assert kb_search.annotations is not None
    assert kb_search.annotations.read_only_hint is True
    assert kb_search.annotations.idempotent_hint is True
    assert kb_search.annotations.open_world_hint is False


@pytest.mark.asyncio
async def test_mcp_kb_search_calls_existing_tool_and_returns_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, str | None]] = []
    events: list[dict[str, Any]] = []

    def fake_kb_search(
        query: str,
        top_n: int,
        *,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        calls.append((query, top_n, trace_id))
        return {
            "hits": [
                {
                    "chunk_id": "chunk-1",
                    "text": "LangGraph 使用 StateGraph 描述状态机。",
                    "source": "data/kb/langgraph.md",
                    "chunk_index": 0,
                    "ranking_score": 0.75,
                    "score": 0.75,
                    "score_kind": "rrf",
                    "fallback_confidence": 0.91,
                    "channel": "hybrid",
                }
            ],
            "max_score": 0.75,
            "fallback_confidence": 0.91,
            "fallback_confidence_kind": "vector_cosine",
        }

    monkeypatch.setattr(server_module, "_kb_search", fake_kb_search)
    monkeypatch.setattr(server_module, "new_trace_id", lambda: "mcp-kb-trace")
    monkeypatch.setattr(server_module, "emit", lambda event: events.append(dict(event)))

    async with Client(server_module.server) as client:
        response = await client.call_tool(
            "kb_search",
            {"query": "  LangGraph 状态机  ", "top_n": 3},
        )

    assert response.is_error is False
    assert calls == [("LangGraph 状态机", 3, "mcp-kb-trace")]
    assert response.structured_content == {
        "query": "LangGraph 状态机",
        "trace_id": "mcp-kb-trace",
        "hits": [
            {
                "chunk_id": "chunk-1",
                "text": "LangGraph 使用 StateGraph 描述状态机。",
                "source": "data/kb/langgraph.md",
                "chunk_index": 0,
                "ranking_score": 0.75,
                "score": 0.75,
                "score_kind": "rrf",
                "fallback_confidence": 0.91,
                "channel": "hybrid",
            }
        ],
        "max_score": 0.75,
        "fallback_confidence": 0.91,
        "fallback_confidence_kind": "vector_cosine",
    }
    assert [event["event"] for event in events] == ["task_start", "task_end"]


@pytest.mark.asyncio
async def test_mcp_deep_research_adapts_langgraph_terminal_event(
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
        yield {
            "event": "complete",
            "data": json.dumps(
                {
                    "status": "completed",
                    "thread_id": "thread-123",
                    "trace_id": "thread-123",
                    "state": {
                        "topic": topic,
                        "final_report": "# LangGraph 研究报告",
                        "research_result_count": 3,
                        "citation_count": 4,
                        "quality_score": 0.82,
                        "revision_count": 1,
                        "fallback_queries": ["MCP protocol"],
                        "errors": [],
                        "usage": {
                            "prompt_tokens": 120,
                            "completion_tokens": 80,
                            "total_tokens": 200,
                            "total_cost": 0.003,
                            "llm_calls": 4,
                            "total_latency_ms": 1500.5,
                        },
                    },
                },
                ensure_ascii=False,
            ),
        }

    monkeypatch.setattr(server_module, "stream_research_progress", fake_stream)

    async with Client(server_module.server) as client:
        response = await client.call_tool(
            "deep_research",
            {"topic": "  LangGraph 与 MCP 结合  "},
        )

    assert response.is_error is False
    assert calls == [("LangGraph 与 MCP 结合", None, False)]
    assert response.structured_content["status"] == "completed"
    assert response.structured_content["final_report"] == "# LangGraph 研究报告"
    assert response.structured_content["thread_id"] == "thread-123"
    assert response.structured_content["quality_score"] == 0.82
    assert response.structured_content["usage"]["total_tokens"] == 200


@pytest.mark.asyncio
async def test_mcp_schema_rejects_invalid_arguments_before_tool_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fail_if_called(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        raise AssertionError("kb_search should not be called")

    monkeypatch.setattr(server_module, "_kb_search", fail_if_called)

    async with Client(server_module.server) as client:
        blank_query = await client.call_tool("kb_search", {"query": "   "})
        invalid_top_n = await client.call_tool(
            "kb_search",
            {"query": "LangGraph", "top_n": 21},
        )
        invalid_thread_id = await client.call_tool(
            "deep_research",
            {"topic": "LangGraph", "thread_id": "../unsafe"},
        )

    assert blank_query.is_error is True
    assert invalid_top_n.is_error is True
    assert invalid_thread_id.is_error is True
    assert called is False


@pytest.mark.asyncio
async def test_mcp_resume_requires_thread_id() -> None:
    async with Client(server_module.server) as client:
        response = await client.call_tool(
            "deep_research",
            {"topic": "LangGraph", "resume": True},
        )

    assert response.is_error is True
    assert "thread_id is required" in response.content[0].text


def test_mcp_main_uses_stdio_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    transports: list[str] = []
    monkeypatch.setattr(
        server_module.server,
        "run",
        lambda *, transport: transports.append(transport),
    )

    server_module.main()

    assert transports == ["stdio"]
