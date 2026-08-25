"""Critic Agent：只评审研究资料质量并指出定向补查缺口。"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Any

from langgraph.types import StreamWriter

from agents.prompt_loader import load_prompt
from agents.state import ResearchState
from core.llm import LLMError, achat
from core.trace import emit, new_trace_id

logger = logging.getLogger(__name__)
NEUTRAL_QUALITY_SCORE = 0.5


def _no_op_stream_writer(_: Any) -> None:
    return None


async def critic_node(
    state: ResearchState,
    *,
    writer: StreamWriter = _no_op_stream_writer,
) -> dict[str, object]:
    """评审已有资料；结构化解析失败时降级，不中断主流程。"""
    trace_id = state.get("trace_id", "") or new_trace_id()
    started_at = time.perf_counter()
    research_results = state.get("research_results", {})
    errors = list(state.get("errors", []))
    logger.info("critic_node enter research_results=%s", len(research_results))
    emit(
        {
            "trace_id": trace_id,
            "event": "node_start",
            "node": "critic",
            "payload": {"research_result_count": len(research_results)},
        }
    )
    _write_custom_event(
        writer,
        "critic_start",
        {"node": "critic", "research_result_count": len(research_results)},
    )

    try:
        if not research_results:
            raise ValueError("research_results must not be empty")
        result = await achat(
            [
                {"role": "system", "content": load_prompt("critic_system")},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "topic": state.get("topic", ""),
                            "sub_questions": state.get("sub_questions", []),
                            "research_results": research_results,
                            "citations": state.get("citations", {}),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            node="critic",
            trace_id=trace_id,
            json_mode=True,
        )
        quality_score, critique, missing_aspects = _parse_critique(result.content)
        quality_history = [*state.get("quality_history", []), quality_score]
        latency_ms = (time.perf_counter() - started_at) * 1000
        emit(
            {
                "trace_id": trace_id,
                "event": "node_end",
                "node": "critic",
                "payload": {
                    "status": "completed",
                    "quality_score": quality_score,
                    "missing_aspect_count": len(missing_aspects),
                    "latency_ms": latency_ms,
                },
            }
        )
        _write_custom_event(
            writer,
            "critic_done",
            {
                "node": "critic",
                "quality_score": quality_score,
                "critique": critique,
                "missing_aspects": missing_aspects,
            },
        )
        logger.info(
            "critic_node output quality_score=%.3f missing_aspects=%s",
            quality_score,
            len(missing_aspects),
        )
        return {
            "quality_score": quality_score,
            "quality_history": quality_history,
            "critique": critique,
            "missing_aspects": missing_aspects,
        }
    except Exception as exc:
        normalized_exc = _normalize_error(exc)
        errors.append(f"Critic: {normalized_exc}")
        critique = f"Critic 输出不可用，已按中性分数降级：{normalized_exc}"
        quality_history = [
            *state.get("quality_history", []),
            NEUTRAL_QUALITY_SCORE,
        ]
        latency_ms = (time.perf_counter() - started_at) * 1000
        logger.error("critic_node degraded: %s", normalized_exc)
        emit(
            {
                "trace_id": trace_id,
                "event": "error",
                "node": "critic",
                "payload": {
                    "type": type(exc).__name__,
                    "message": str(normalized_exc),
                },
            }
        )
        emit(
            {
                "trace_id": trace_id,
                "event": "node_end",
                "node": "critic",
                "payload": {
                    "status": "degraded",
                    "quality_score": NEUTRAL_QUALITY_SCORE,
                    "missing_aspect_count": 0,
                    "latency_ms": latency_ms,
                },
            }
        )
        _write_custom_event(
            writer,
            "critic_done",
            {
                "node": "critic",
                "quality_score": NEUTRAL_QUALITY_SCORE,
                "critique": critique,
                "missing_aspects": [],
                "degraded": True,
            },
        )
        return {
            "quality_score": NEUTRAL_QUALITY_SCORE,
            "quality_history": quality_history,
            "critique": critique,
            "missing_aspects": [],
            "errors": errors,
        }


def _parse_critique(content: str) -> tuple[float, str, list[str]]:
    try:
        payload = json.loads(_strip_json_fence(content))
    except json.JSONDecodeError as exc:
        raise ValueError("structured output is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("structured output must be a JSON object")

    try:
        raw_score = payload["quality_score"]
        if isinstance(raw_score, bool):
            raise TypeError
        quality_score = float(raw_score)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("quality_score must be a number") from exc
    if not math.isfinite(quality_score):
        raise ValueError("quality_score must be a finite number")
    quality_score = max(0.0, min(1.0, quality_score))

    critique = str(payload.get("critique", "")).strip()
    if not critique:
        raise ValueError("critique must not be empty")
    raw_missing = payload.get("missing_aspects", [])
    if not isinstance(raw_missing, list):
        raise ValueError("missing_aspects must be a list")
    missing_aspects = _normalize_aspects(raw_missing)
    return quality_score, critique, missing_aspects


def _normalize_aspects(raw_aspects: list[Any]) -> list[str]:
    aspects: list[str] = []
    seen: set[str] = set()
    for raw_aspect in raw_aspects:
        if not isinstance(raw_aspect, str):
            continue
        aspect = raw_aspect.strip()
        if not aspect or aspect in seen:
            continue
        seen.add(aspect)
        aspects.append(aspect)
    return aspects


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _normalize_error(exc: Exception) -> Exception:
    if isinstance(exc, LLMError):
        return RuntimeError(str(exc))
    return exc


def _write_custom_event(
    writer: StreamWriter,
    event: str,
    payload: dict[str, Any],
) -> None:
    try:
        writer({"event": event, "payload": payload})
    except RuntimeError:
        return


__all__ = ["critic_node"]
