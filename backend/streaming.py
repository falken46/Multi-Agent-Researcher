"""后端异步 SSE 推送、trace 用量汇总与 checkpoint 恢复。"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any, cast

from agents.graph import build_graph, create_initial_state
from agents.state import ResearchState, Usage
from core.checkpoint import open_sqlite_checkpointer
from core.trace import emit, summarize

logger = logging.getLogger(__name__)
_MISSING = object()


async def stream_research_progress(
    topic: str,
    compiled_graph: Any | None = None,
    *,
    thread_id: str | None = None,
    resume: bool = False,
) -> AsyncIterator[dict[str, str]]:
    """运行或恢复 LangGraph，并把状态更新转换成 SSE 事件。"""
    normalized_topic = topic.strip()
    if not normalized_topic:
        raise ValueError("topic must not be empty")
    if resume and not thread_id:
        raise ValueError("thread_id is required when resume is true")

    current_state = create_initial_state(normalized_topic, trace_id=thread_id)
    run_thread_id = thread_id or current_state["trace_id"]

    try:
        if compiled_graph is None:
            async with open_sqlite_checkpointer() as checkpointer:
                persistent_graph = build_graph(checkpointer=checkpointer)
                async for event in _stream_with_graph(
                    persistent_graph,
                    current_state=current_state,
                    thread_id=run_thread_id,
                    resume=resume,
                ):
                    yield event
        else:
            async for event in _stream_with_graph(
                compiled_graph,
                current_state=current_state,
                thread_id=run_thread_id,
                resume=resume,
            ):
                yield event
    except Exception as exc:
        trace_id = current_state.get("trace_id", "") or run_thread_id
        logger.exception("research stream failed thread_id=%s", run_thread_id)
        emit(
            {
                "trace_id": trace_id,
                "event": "error",
                "node": "backend",
                "payload": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        emit(
            {
                "trace_id": trace_id,
                "event": "task_end",
                "payload": {"status": "failed", "thread_id": run_thread_id},
            }
        )
        yield _sse_event(
            event="error",
            payload={
                "node": "error",
                "status": "failed",
                "error": str(exc),
                "thread_id": run_thread_id,
                "trace_id": trace_id,
                "state": _state_summary(current_state),
            },
        )


async def _stream_with_graph(
    compiled_graph: Any,
    *,
    current_state: ResearchState,
    thread_id: str,
    resume: bool,
) -> AsyncIterator[dict[str, str]]:
    config = {"configurable": {"thread_id": thread_id}}
    graph_input: ResearchState | None = current_state
    if resume:
        snapshot = await compiled_graph.aget_state(config)
        saved_values = getattr(snapshot, "values", None)
        if not isinstance(saved_values, dict) or not saved_values:
            raise ValueError(f"checkpoint not found for thread_id={thread_id}")
        saved_topic = str(saved_values.get("topic", "")).strip()
        if saved_topic and saved_topic != current_state["topic"]:
            raise ValueError("topic does not match the saved checkpoint")
        restored_state = create_initial_state(
            saved_topic or current_state["topic"],
            trace_id=str(saved_values.get("trace_id", "")) or thread_id,
        )
        restored_state.update(cast(ResearchState, saved_values))
        current_state.clear()
        current_state.update(restored_state)
        graph_input = None

    trace_id = current_state.get("trace_id", "") or thread_id
    logger.info(
        "research stream start topic=%r thread_id=%s resume=%s",
        current_state["topic"][:100],
        thread_id,
        resume,
    )
    emit(
        {
            "trace_id": trace_id,
            "event": "task_resume" if resume else "task_start",
            "payload": {
                "topic": current_state["topic"],
                "thread_id": thread_id,
                "resume": resume,
            },
        }
    )
    yield _sse_event(
        event="start",
        payload={
            "node": "start",
            "status": "resumed" if resume else "started",
            "topic": current_state["topic"],
            "thread_id": thread_id,
            "trace_id": trace_id,
            "resumed": resume,
        },
    )

    stream_options: dict[str, Any] = {
        "stream_mode": ["updates", "custom"],
        "version": "v2",
    }
    # LangGraph 1.1.x 在“无 checkpointer + durability=sync”时会触发内部
    # checkpoint future 错误；生产图有 SQLite saver，注入的纯测试图则省略该参数。
    if getattr(compiled_graph, "checkpointer", _MISSING) is not None:
        stream_options["durability"] = "sync"
    async for part in compiled_graph.astream(
        graph_input,
        config,
        **stream_options,
    ):
        stream_type, stream_data = _unpack_stream_part(part)
        if stream_type == "custom":
            custom_event = _custom_sse_event(stream_data)
            if custom_event is not None:
                yield custom_event
            continue
        if stream_type != "updates" or not isinstance(stream_data, dict):
            continue

        for node_name, node_update in stream_data.items():
            if node_name == "__interrupt__":
                continue
            previous_errors = list(current_state.get("errors", []))
            if isinstance(node_update, dict):
                current_state.update(cast(ResearchState, node_update))

            logger.info("research stream node=%s thread_id=%s", node_name, thread_id)
            yield _sse_event(
                event="progress",
                payload={
                    "node": node_name,
                    "status": _node_status(
                        node_name=node_name,
                        node_update=node_update,
                        previous_errors=previous_errors,
                        state=current_state,
                    ),
                    "state": _state_summary(current_state),
                    "update": _safe_json_value(node_update),
                    "thread_id": thread_id,
                    "trace_id": trace_id,
                },
            )

    final_status = _final_status(current_state)
    emit(
        {
            "trace_id": trace_id,
            "event": "task_end",
            "payload": {"status": final_status, "thread_id": thread_id},
        }
    )
    trace_summary = summarize(trace_id)
    usage = _usage_from_trace(trace_summary)
    current_state["usage"] = usage
    yield _sse_event(
        event="usage",
        payload={
            "node": "usage",
            "status": "completed",
            "usage": usage,
            "fallback_count": trace_summary.get("fallback_count", 0),
            "revision_count": trace_summary.get("revision_count", 0),
            "thread_id": thread_id,
            "trace_id": trace_id,
        },
    )
    yield _sse_event(
        event="complete",
        payload={
            "node": "end",
            "status": final_status,
            "state": _state_summary(current_state),
            "trace": trace_summary,
            "thread_id": thread_id,
            "trace_id": trace_id,
        },
    )
    logger.info("research stream complete thread_id=%s status=%s", thread_id, final_status)


def _unpack_stream_part(part: Any) -> tuple[str, Any]:
    """兼容 LangGraph v2 字典事件及测试中常用的二元组事件。"""
    if isinstance(part, dict) and "type" in part:
        return str(part.get("type", "")), part.get("data")
    if isinstance(part, tuple) and len(part) == 2:
        return str(part[0]), part[1]
    if isinstance(part, dict):
        return "updates", part
    return "", None


def _custom_sse_event(data: Any) -> dict[str, str] | None:
    if not isinstance(data, dict):
        return None
    event_name = str(data.get("event", "")).strip()
    payload = data.get("payload", {})
    if not event_name or not isinstance(payload, dict):
        return None
    return _sse_event(event=event_name, payload=payload)


def _sse_event(event: str, payload: dict[str, Any]) -> dict[str, str]:
    return {
        "event": event,
        "data": json.dumps(payload, ensure_ascii=False),
    }


def _state_summary(state: ResearchState) -> dict[str, Any]:
    return {
        "topic": state.get("topic", ""),
        "sub_questions": state.get("sub_questions", []),
        "research_result_count": len(state.get("research_results", {})),
        "citation_count": sum(
            len(items) for items in state.get("citations", {}).values()
        ),
        "final_report": state.get("final_report", ""),
        "errors": state.get("errors", []),
        "retry_count": state.get("retry_count", 0),
        "critique": state.get("critique", ""),
        "quality_score": state.get("quality_score", 0.0),
        "quality_history": state.get("quality_history", []),
        "missing_aspects": state.get("missing_aspects", []),
        "revision_count": state.get("revision_count", 0),
        "fallback_queries": state.get("fallback_queries", []),
        "usage": state.get("usage", {}),
        "trace_id": state.get("trace_id", ""),
    }


def _usage_from_trace(summary: dict[str, Any]) -> Usage:
    return {
        "prompt_tokens": _safe_int(summary.get("prompt_tokens")),
        "completion_tokens": _safe_int(summary.get("completion_tokens")),
        "total_tokens": _safe_int(summary.get("total_tokens")),
        "total_cost": _safe_float(summary.get("total_cost")),
        "llm_calls": _safe_int(summary.get("llm_calls")),
        "total_latency_ms": _safe_float(summary.get("total_latency_ms")),
    }


def _node_status(
    node_name: str,
    node_update: Any,
    previous_errors: list[str],
    state: ResearchState,
) -> str:
    if _has_new_errors(node_update=node_update, previous_errors=previous_errors):
        if node_name in {"researcher", "critic"} and state.get("research_results"):
            return "warning"
        return "failed"
    return "completed"


def _final_status(state: ResearchState) -> str:
    if state.get("final_report"):
        return "completed"
    if state.get("errors"):
        return "failed"
    return "empty"


def _has_new_errors(node_update: Any, previous_errors: list[str]) -> bool:
    if not isinstance(node_update, dict):
        return False
    updated_errors = node_update.get("errors")
    if not isinstance(updated_errors, list):
        return False
    return len(updated_errors) > len(previous_errors)


def _safe_json_value(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return str(value)


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = ["stream_research_progress"]
