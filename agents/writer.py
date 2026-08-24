"""Writer Agent 节点实现。"""

from __future__ import annotations

import logging
import time

from agents.prompt_loader import load_prompt
from agents.state import ResearchState
from core.llm import LLMError, chat
from core.trace import emit, new_trace_id

logger = logging.getLogger(__name__)

class WriterError(RuntimeError):
    """Writer Agent 写作失败时抛出。"""


def writer_node(state: ResearchState) -> dict[str, object]:
    """根据研究资料生成 Markdown 报告。"""
    topic = state.get("topic", "").strip()
    sub_questions = state.get("sub_questions", [])
    research_results = state.get("research_results", {})
    errors = list(state.get("errors", []))
    trace_id = state.get("trace_id", "") or new_trace_id()
    started_at = time.perf_counter()
    logger.info(
        "writer_node enter topic=%r sub_questions=%s results=%s errors=%s",
        topic[:100],
        len(sub_questions),
        len(research_results),
        len(errors),
    )
    emit(
        {
            "trace_id": trace_id,
            "event": "node_start",
            "node": "writer",
            "payload": {
                "sub_question_count": len(sub_questions),
                "research_result_count": len(research_results),
            },
        }
    )

    try:
        if not topic:
            raise WriterError("topic must not be empty")
        if not sub_questions:
            raise WriterError("sub_questions must not be empty")
        if not research_results:
            raise WriterError("research_results must not be empty")

        system_prompt = load_prompt("writer_system")
        user_prompt = build_writer_prompt(
            topic=topic,
            sub_questions=sub_questions,
            research_results=research_results,
            errors=errors,
        )
        final_report = _call_writer_model(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            trace_id=trace_id,
        )
        logger.info("writer_node output final_report_chars=%s", len(final_report))
        emit(
            {
                "trace_id": trace_id,
                "event": "node_end",
                "node": "writer",
                "payload": {
                    "status": "completed",
                    "final_report_chars": len(final_report),
                    "latency_ms": (time.perf_counter() - started_at) * 1000,
                },
            }
        )
        return {"final_report": final_report}
    except Exception as exc:
        logger.error("writer_node failed: %s", exc)
        errors.append(f"Writer: {exc}")
        _emit_node_error(trace_id, exc, started_at)
        return {"errors": errors}


def build_writer_prompt(
    topic: str,
    sub_questions: list[str],
    research_results: dict[str, str],
    errors: list[str] | None = None,
) -> str:
    """拼装 Writer Agent 的用户输入 prompt。"""
    question_blocks = []
    for index, question in enumerate(sub_questions, start=1):
        result = research_results.get(question, "资料不足或未检索到结果。")
        question_blocks.append(
            "\n".join(
                [
                    f"### 子问题 {index}",
                    f"问题: {question}",
                    "资料摘要:",
                    result,
                ]
            )
        )

    error_block = "\n".join(f"- {error}" for error in errors or [])
    if not error_block:
        error_block = "无"

    return "\n\n".join(
        [
            f"研究主题:\n{topic}",
            "子问题与资料:",
            "\n\n".join(question_blocks),
            "流程错误记录:",
            error_block,
            "请基于以上资料生成完整 Markdown 研究报告。",
        ]
    )


def _call_writer_model(user_prompt: str, system_prompt: str, trace_id: str) -> str:
    try:
        return chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            node="writer",
            trace_id=trace_id,
        ).content
    except LLMError as exc:
        raise WriterError(str(exc)) from exc


def _emit_node_error(trace_id: str, exc: Exception, started_at: float) -> None:
    emit(
        {
            "trace_id": trace_id,
            "event": "error",
            "node": "writer",
            "payload": {"type": type(exc).__name__, "message": str(exc)},
        }
    )
    emit(
        {
            "trace_id": trace_id,
            "event": "node_end",
            "node": "writer",
            "payload": {
                "status": "failed",
                "latency_ms": (time.perf_counter() - started_at) * 1000,
            },
        }
    )


__all__ = ["build_writer_prompt", "writer_node"]
