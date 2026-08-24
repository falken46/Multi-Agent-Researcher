"""后端 SSE 流式推送实现。"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from agents.graph import create_initial_state, graph
from agents.state import ResearchState

logger = logging.getLogger(__name__)


def stream_research_progress(
    topic: str,
    compiled_graph: Any = graph,
) -> Iterator[dict[str, str]]:
    """运行 LangGraph,按节点状态变更输出 SSE 事件。"""
    normalized_topic = topic.strip()
    current_state: ResearchState = create_initial_state(normalized_topic)

    logger.info("research stream start topic=%r", normalized_topic[:100])
    yield _sse_event(
        event="start",
        payload={
            "node": "start",
            "status": "started",
            "topic": normalized_topic,
        },
    )

    try:
        for update in compiled_graph.stream(current_state, stream_mode="updates"):
            for node_name, node_update in update.items():
                previous_errors = list(current_state.get("errors", []))
                if isinstance(node_update, dict):
                    current_state.update(node_update)

                logger.info("research stream node=%s", node_name)
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
                    },
                )

        yield _sse_event(
            event="complete",
            payload={
                "node": "end",
                "status": _final_status(current_state),
                "state": _state_summary(current_state),
            },
        )
        logger.info("research stream complete topic=%r", normalized_topic[:100])
    except Exception as exc:
        logger.error("research stream failed: %s", exc)
        yield _sse_event(
            event="error",
            payload={
                "node": "error",
                "status": "failed",
                "error": str(exc),
                "state": _state_summary(current_state),
            },
        )


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
        "final_report": state.get("final_report", ""),
        "errors": state.get("errors", []),
        "retry_count": state.get("retry_count", 0),
    }


def _node_status(
    node_name: str,
    node_update: Any,
    previous_errors: list[str],
    state: ResearchState,
) -> str:
    if _has_new_errors(node_update=node_update, previous_errors=previous_errors):
        if node_name == "researcher" and state.get("research_results"):
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


__all__ = ["stream_research_progress"]
