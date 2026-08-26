"""通过官方 MCP Python SDK 暴露研究工作流与本地知识库检索。"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, BeforeValidator, Field, StringConstraints

from backend.streaming import stream_research_progress
from core.trace import emit, new_trace_id
from tools.kb_search import kb_search as _kb_search

logger = logging.getLogger(__name__)

SERVER_NAME = "deepresearch-agent"
SERVER_VERSION = "0.1.0"
_TRACE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$"


def _strip_text(value: object) -> object:
    """让输入 schema 在长度校验前先去除首尾空白。"""
    return value.strip() if isinstance(value, str) else value


ResearchTopic = Annotated[
    str,
    BeforeValidator(_strip_text),
    Field(
        min_length=1,
        max_length=500,
        description="需要完整研究并生成 Markdown 报告的主题。",
    ),
]
ThreadIdValue = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=_TRACE_ID_PATTERN,
    ),
]
ThreadId = Annotated[
    ThreadIdValue | None,
    Field(
        default=None,
        description="可选任务 ID；恢复任务时必须传入此前返回的 thread_id。",
    ),
]
SearchQuery = Annotated[
    str,
    BeforeValidator(_strip_text),
    Field(
        min_length=1,
        max_length=500,
        description="要在本地知识库中检索的具体问题或关键词。",
    ),
]
TopN = Annotated[
    int,
    Field(
        ge=1,
        le=20,
        description="最多返回多少条结果，范围为 1 到 20。",
    ),
]


class UsageResult(BaseModel):
    """一次研究任务从 trace 汇总得到的资源消耗。"""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    total_cost: float = Field(default=0.0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    total_latency_ms: float = Field(default=0.0, ge=0)


class DeepResearchResult(BaseModel):
    """完整 LangGraph 研究工作流的结构化结果。"""

    status: Literal["completed", "failed", "empty"]
    topic: str
    thread_id: str
    trace_id: str
    resumed: bool
    final_report: str
    research_result_count: int = Field(ge=0)
    citation_count: int = Field(ge=0)
    quality_score: float = Field(ge=0, le=1)
    revision_count: int = Field(ge=0)
    fallback_queries: list[str]
    errors: list[str]
    usage: UsageResult


class KBSearchHitResult(BaseModel):
    """一条本地知识库命中。"""

    chunk_id: str
    text: str
    source: str
    chunk_index: int = Field(ge=0)
    ranking_score: float
    score: float
    score_kind: str
    fallback_confidence: float | None
    channel: str


class KBSearchResult(BaseModel):
    """本地知识库检索的结构化结果与降级置信度。"""

    query: str
    trace_id: str
    hits: list[KBSearchHitResult]
    max_score: float
    fallback_confidence: float = Field(ge=0, le=1)
    fallback_confidence_kind: str


server = MCPServer(
    name=SERVER_NAME,
    title="DeepResearch Agent",
    description="基于 LangGraph 的多智能体研究助手 MCP Server。",
    instructions=(
        "需要完整研究报告时调用 deep_research；只需要查询项目本地知识库时调用 "
        "kb_search。deep_research 可能调用 LLM 与联网搜索并产生费用；kb_search 只读本地索引。"
    ),
    version=SERVER_VERSION,
)


@server.tool(
    name="deep_research",
    title="执行深度研究",
    description=(
        "运行完整的 Planner → Researcher ⇄ Critic → Writer LangGraph 工作流，"
        "返回 Markdown 报告、质量状态、trace 用量和可恢复的 thread_id。"
    ),
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
    structured_output=True,
)
async def deep_research_tool(
    topic: ResearchTopic,
    thread_id: ThreadId = None,
    resume: Annotated[
        bool,
        Field(
            description=(
                "是否从 SQLite checkpoint 恢复；为 true 时必须同时提供 thread_id，"
                "且 topic 必须与原任务一致。"
            )
        ),
    ] = False,
) -> DeepResearchResult:
    """执行或恢复一次完整研究任务。"""
    if resume and thread_id is None:
        raise ToolError("thread_id is required when resume is true")

    logger.info(
        "mcp deep_research input topic=%r thread_id=%s resume=%s",
        topic[:100],
        thread_id or "",
        resume,
    )
    terminal_payload: dict[str, Any] | None = None
    terminal_event = ""
    async for event in stream_research_progress(
        topic,
        thread_id=thread_id,
        resume=resume,
    ):
        event_name = str(event.get("event", ""))
        if event_name not in {"complete", "error"}:
            continue
        terminal_event = event_name
        terminal_payload = _decode_event_payload(event)

    if terminal_payload is None:
        raise RuntimeError("research workflow ended without a terminal event")

    result = _deep_research_result(
        terminal_payload,
        terminal_event=terminal_event,
        requested_topic=topic,
        requested_thread_id=thread_id,
        resumed=resume,
    )
    logger.info(
        "mcp deep_research output status=%s thread_id=%s report_chars=%s",
        result.status,
        result.thread_id,
        len(result.final_report),
    )
    return result


@server.tool(
    name="kb_search",
    title="查询本地知识库",
    description=(
        "只读查询本地 Chroma + BM25 + RRF 检索流水线，返回命中文本、来源、排序分和"
        "用于判断是否需要联网降级的独立置信度；不会调用 LLM 或 Web 搜索。"
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    structured_output=True,
)
async def kb_search_tool(
    query: SearchQuery,
    top_n: TopN = 5,
) -> KBSearchResult:
    """查询本地知识库，不在 MCP 层决定是否联网降级。"""
    trace_id = new_trace_id()
    logger.info("mcp kb_search input query=%r top_n=%s", query[:100], top_n)
    emit(
        {
            "trace_id": trace_id,
            "event": "task_start",
            "node": "mcp",
            "payload": {"tool": "kb_search", "query": query, "top_n": top_n},
        }
    )
    try:
        raw_result = await asyncio.to_thread(
            _kb_search,
            query,
            top_n,
            trace_id=trace_id,
        )
    except Exception as exc:
        emit(
            {
                "trace_id": trace_id,
                "event": "error",
                "node": "mcp",
                "payload": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        emit(
            {
                "trace_id": trace_id,
                "event": "task_end",
                "node": "mcp",
                "payload": {"tool": "kb_search", "status": "failed"},
            }
        )
        raise

    emit(
        {
            "trace_id": trace_id,
            "event": "task_end",
            "node": "mcp",
            "payload": {
                "tool": "kb_search",
                "status": "completed",
                "hit_count": len(raw_result["hits"]),
            },
        }
    )
    result = KBSearchResult(
        query=query,
        trace_id=trace_id,
        hits=[KBSearchHitResult.model_validate(hit) for hit in raw_result["hits"]],
        max_score=raw_result["max_score"],
        fallback_confidence=raw_result["fallback_confidence"],
        fallback_confidence_kind=raw_result["fallback_confidence_kind"],
    )
    logger.info(
        "mcp kb_search output hits=%s trace_id=%s",
        len(result.hits),
        trace_id,
    )
    return result


def _decode_event_payload(event: dict[str, str]) -> dict[str, Any]:
    try:
        payload = json.loads(event.get("data", ""))
    except json.JSONDecodeError as exc:
        raise RuntimeError("research workflow emitted invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("research workflow terminal payload must be an object")
    return payload


def _deep_research_result(
    payload: dict[str, Any],
    *,
    terminal_event: str,
    requested_topic: str,
    requested_thread_id: str | None,
    resumed: bool,
) -> DeepResearchResult:
    state = payload.get("state")
    if not isinstance(state, dict):
        state = {}
    raw_errors = state.get("errors")
    errors = [str(item) for item in raw_errors] if isinstance(raw_errors, list) else []
    terminal_error = str(payload.get("error", "")).strip()
    if terminal_error and terminal_error not in errors:
        errors.append(terminal_error)

    raw_status = str(payload.get("status", "")).strip().lower()
    status: Literal["completed", "failed", "empty"]
    if terminal_event == "error" or raw_status == "failed":
        status = "failed"
    elif raw_status == "completed":
        status = "completed"
    else:
        status = "empty"

    usage = state.get("usage")
    usage_payload = usage if isinstance(usage, dict) else {}
    thread_id = str(payload.get("thread_id", "")).strip() or requested_thread_id or ""
    trace_id = str(payload.get("trace_id", "")).strip() or thread_id
    if not thread_id or not trace_id:
        raise RuntimeError("research workflow terminal payload lacks task identity")

    return DeepResearchResult(
        status=status,
        topic=str(state.get("topic", "")).strip() or requested_topic,
        thread_id=thread_id,
        trace_id=trace_id,
        resumed=resumed,
        final_report=str(state.get("final_report", "")),
        research_result_count=_non_negative_int(state.get("research_result_count")),
        citation_count=_non_negative_int(state.get("citation_count")),
        quality_score=_bounded_score(state.get("quality_score")),
        revision_count=_non_negative_int(state.get("revision_count")),
        fallback_queries=_string_list(state.get("fallback_queries")),
        errors=errors,
        usage=UsageResult.model_validate(usage_payload),
    )


def _non_negative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _bounded_score(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def main() -> None:
    """以本地 stdio transport 启动 MCP Server。"""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()


__all__ = [
    "DeepResearchResult",
    "KBSearchResult",
    "SERVER_NAME",
    "SERVER_VERSION",
    "deep_research_tool",
    "kb_search_tool",
    "main",
    "server",
]
