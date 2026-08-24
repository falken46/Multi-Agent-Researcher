"""Researcher Agent 节点实现。"""

from __future__ import annotations

import logging
import time

from agents.prompt_loader import load_prompt
from agents.state import ResearchState
from core.llm import LLMError, chat
from core.trace import emit, new_trace_id
from tools.web_search import SearchResult, web_search

logger = logging.getLogger(__name__)

MAX_SEARCH_RESULTS = 3


class ResearcherError(RuntimeError):
    """Researcher Agent 研究失败时抛出。"""


def researcher_node(state: ResearchState) -> dict[str, object]:
    """遍历研究子问题,返回每个子问题对应的资料摘要。"""
    sub_questions = state.get("sub_questions", [])
    trace_id = state.get("trace_id", "") or new_trace_id()
    started_at = time.perf_counter()
    logger.info("researcher_node enter sub_questions=%s", len(sub_questions))
    emit(
        {
            "trace_id": trace_id,
            "event": "node_start",
            "node": "researcher",
            "payload": {"sub_question_count": len(sub_questions)},
        }
    )

    research_results: dict[str, str] = dict(state.get("research_results", {}))
    errors = list(state.get("errors", []))

    if not sub_questions:
        exc = ResearcherError("sub_questions must not be empty")
        errors.append(f"Researcher: {exc}")
        _emit_node_error(trace_id, exc, started_at)
        return {"errors": errors}

    try:
        system_prompt = load_prompt("researcher_system")
    except Exception as exc:
        logger.error("researcher_node failed: %s", exc)
        errors.append(f"Researcher: {exc}")
        _emit_node_error(trace_id, exc, started_at)
        return {"errors": errors}

    for question in sub_questions:
        try:
            summary = _research_question(
                question=question,
                system_prompt=system_prompt,
                trace_id=trace_id,
            )
            research_results[question] = summary
            logger.info("researcher_node question done chars=%s", len(summary))
        except Exception as exc:
            logger.error("researcher_node question failed: %s", exc)
            errors.append(f"Researcher: {question} | {exc}")
            emit(
                {
                    "trace_id": trace_id,
                    "event": "error",
                    "node": "researcher",
                    "payload": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "question": question[:200],
                    },
                }
            )

    logger.info(
        "researcher_node output research_results=%s errors=%s",
        len(research_results),
        len(errors),
    )
    emit(
        {
            "trace_id": trace_id,
            "event": "node_end",
            "node": "researcher",
            "payload": {
                "status": "completed" if research_results else "failed",
                "research_result_count": len(research_results),
                "error_count": len(errors),
                "latency_ms": (time.perf_counter() - started_at) * 1000,
            },
        }
    )
    result: dict[str, object] = {
        "research_results": research_results,
        "errors": errors,
    }
    if sub_questions and not research_results:
        result["retry_count"] = state.get("retry_count", 0) + 1
    return result


def _research_question(question: str, system_prompt: str, trace_id: str) -> str:
    normalized_question = question.strip()
    if not normalized_question:
        raise ResearcherError("question must not be empty")

    search_results = web_search(normalized_question, max_results=MAX_SEARCH_RESULTS)
    summary = _call_summary_model(
        question=normalized_question,
        search_results=search_results,
        system_prompt=system_prompt,
        trace_id=trace_id,
    )
    return _append_sources(summary=summary, search_results=search_results)


def _call_summary_model(
    question: str,
    search_results: list[SearchResult],
    system_prompt: str,
    trace_id: str,
) -> str:
    try:
        return chat(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": _build_summary_prompt(question, search_results),
                },
            ],
            node="researcher",
            trace_id=trace_id,
        ).content
    except LLMError as exc:
        raise ResearcherError(str(exc)) from exc


def _build_summary_prompt(question: str, search_results: list[SearchResult]) -> str:
    result_blocks = []
    for index, result in enumerate(search_results, start=1):
        result_blocks.append(
            "\n".join(
                [
                    f"{index}. {result['title']}",
                    f"URL: {result['url']}",
                    f"摘要: {result['snippet']}",
                ]
            )
        )

    return "\n\n".join(
        [
            f"子问题:\n{question}",
            "搜索资料:",
            "\n\n".join(result_blocks),
            "请基于以上资料生成资料摘要,并保留来源 URL。",
        ]
    )


def _append_sources(summary: str, search_results: list[SearchResult]) -> str:
    urls = []
    seen_urls = set()
    for result in search_results:
        url = result["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        urls.append(url)

    source_lines = "\n".join(f"- {url}" for url in urls)
    if "来源:" in summary:
        return summary
    return f"{summary}\n\n来源:\n{source_lines}"


def _emit_node_error(trace_id: str, exc: Exception, started_at: float) -> None:
    emit(
        {
            "trace_id": trace_id,
            "event": "error",
            "node": "researcher",
            "payload": {"type": type(exc).__name__, "message": str(exc)},
        }
    )
    emit(
        {
            "trace_id": trace_id,
            "event": "node_end",
            "node": "researcher",
            "payload": {
                "status": "failed",
                "latency_ms": (time.perf_counter() - started_at) * 1000,
            },
        }
    )


__all__ = ["researcher_node"]
