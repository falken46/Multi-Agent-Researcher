"""Planner Agent 节点实现。"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from agents.prompt_loader import load_prompt
from agents.state import ResearchState
from core.llm import LLMError, chat
from core.trace import emit, new_trace_id

logger = logging.getLogger(__name__)

class PlannerError(RuntimeError):
    """Planner Agent 规划失败时抛出。"""


def planner_node(state: ResearchState) -> dict[str, object]:
    """接收研究主题,返回 3-5 个研究子问题。"""
    topic = state.get("topic", "").strip()
    trace_id = state.get("trace_id", "") or new_trace_id()
    started_at = time.perf_counter()
    logger.info("planner_node enter topic=%r", topic[:100])
    emit(
        {
            "trace_id": trace_id,
            "event": "node_start",
            "node": "planner",
            "payload": {"topic_chars": len(topic)},
        }
    )

    try:
        if not topic:
            raise PlannerError("topic must not be empty")

        system_prompt = load_prompt("planner_system")
        raw_content = _call_planner_model(
            topic=topic,
            system_prompt=system_prompt,
            trace_id=trace_id,
        )
        sub_questions = _parse_sub_questions(raw_content)
        logger.info("planner_node output sub_questions=%s", len(sub_questions))
        emit(
            {
                "trace_id": trace_id,
                "event": "node_end",
                "node": "planner",
                "payload": {
                    "status": "completed",
                    "sub_question_count": len(sub_questions),
                    "latency_ms": (time.perf_counter() - started_at) * 1000,
                },
            }
        )
        return {"sub_questions": sub_questions}
    except Exception as exc:
        logger.error("planner_node failed: %s", exc)
        errors = list(state.get("errors", []))
        errors.append(f"Planner: {exc}")
        _emit_node_error(trace_id, exc, started_at)
        return {"errors": errors}


def _call_planner_model(topic: str, system_prompt: str, trace_id: str) -> str:
    try:
        return chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": topic},
            ],
            node="planner",
            trace_id=trace_id,
            json_mode=True,
        ).content
    except LLMError as exc:
        raise PlannerError(str(exc)) from exc


def _emit_node_error(trace_id: str, exc: Exception, started_at: float) -> None:
    emit(
        {
            "trace_id": trace_id,
            "event": "error",
            "node": "planner",
            "payload": {"type": type(exc).__name__, "message": str(exc)},
        }
    )
    emit(
        {
            "trace_id": trace_id,
            "event": "node_end",
            "node": "planner",
            "payload": {
                "status": "failed",
                "latency_ms": (time.perf_counter() - started_at) * 1000,
            },
        }
    )


def _parse_sub_questions(content: str) -> list[str]:
    payload = json.loads(_strip_json_fence(content))
    raw_questions: Any
    if isinstance(payload, list):
        raw_questions = payload
    elif isinstance(payload, dict):
        raw_questions = payload.get("sub_questions")
    else:
        raise PlannerError("planner output must be a JSON object or array")

    if not isinstance(raw_questions, list):
        raise PlannerError("planner output must contain sub_questions list")

    sub_questions = _normalize_questions(raw_questions)
    if len(sub_questions) < 3:
        raise PlannerError("planner output must contain at least 3 sub questions")
    return sub_questions[:5]


def _strip_json_fence(content: str) -> str:
    stripped_content = content.strip()
    if not stripped_content.startswith("```"):
        return stripped_content

    lines = stripped_content.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _normalize_questions(raw_questions: list[Any]) -> list[str]:
    sub_questions: list[str] = []
    seen_questions: set[str] = set()

    for raw_question in raw_questions:
        if not isinstance(raw_question, str):
            continue
        question = raw_question.strip()
        if not question or question in seen_questions:
            continue
        seen_questions.add(question)
        sub_questions.append(question)

    if not sub_questions:
        raise PlannerError("planner output did not contain usable sub questions")
    return sub_questions


__all__ = ["planner_node"]
