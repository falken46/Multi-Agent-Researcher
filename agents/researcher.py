"""Researcher Agent：并发执行本地优先、必要时联网的证据检索。"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.types import StreamWriter

from agents.prompt_loader import load_prompt
from agents.state import Citation, ResearchState
from core.config import Settings, get_settings
from core.llm import LLMError, LLMResult, achat
from core.trace import emit, new_trace_id
from tools.kb_search import KBSearchHit, KBSearchResult, kb_search
from tools.web_search import SearchResult, web_search

logger = logging.getLogger(__name__)
MAX_SEARCH_RESULTS = 3
MAX_KB_RESULTS = 5


class Evidence(TypedDict):
    source: str
    snippet: str
    origin: Literal["kb", "web"]


@dataclass(frozen=True)
class ResearchOutcome:
    target: str
    summary: str
    citations: list[Citation]
    fallback_query: str | None = None


class ResearcherError(RuntimeError):
    """Researcher Agent 研究失败时抛出。"""


def _no_op_stream_writer(_: Any) -> None:
    return None


async def researcher_node(
    state: ResearchState,
    *,
    writer: StreamWriter = _no_op_stream_writer,
) -> dict[str, object]:
    """并发研究普通子问题；返工时只补查 Critic 指出的缺口。"""
    sub_questions = state.get("sub_questions", [])
    missing_aspects = state.get("missing_aspects", [])
    revision_mode = bool(missing_aspects)
    targets = missing_aspects or sub_questions
    trace_id = state.get("trace_id", "") or new_trace_id()
    settings = get_settings()
    started_at = time.perf_counter()
    logger.info(
        "researcher_node enter targets=%s revision_mode=%s concurrency=%s",
        len(targets),
        revision_mode,
        settings.research_concurrency,
    )
    emit(
        {
            "trace_id": trace_id,
            "event": "node_start",
            "node": "researcher",
            "payload": {
                "target_count": len(targets),
                "revision_mode": revision_mode,
                "concurrency": settings.research_concurrency,
            },
        }
    )

    research_results = dict(state.get("research_results", {}))
    citations = {
        key: list(items) for key, items in state.get("citations", {}).items()
    }
    errors = list(state.get("errors", []))
    revision_count = state.get("revision_count", 0)

    if not targets:
        exc = ResearcherError("sub_questions and missing_aspects must not both be empty")
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

    if revision_mode:
        revision_count += 1
        revision_payload = {
            "node": "researcher",
            "round": revision_count,
            "previous_quality_score": state.get("quality_score", 0.0),
            "missing_aspects": list(missing_aspects),
        }
        emit(
            {
                "trace_id": trace_id,
                "event": "revision",
                "node": "researcher",
                "payload": revision_payload,
            }
        )
        _write_custom_event(writer, "revision", revision_payload)

    semaphore = asyncio.Semaphore(settings.research_concurrency)
    timeout_seconds = _research_timeout(settings)
    gathered = await asyncio.gather(
        *(
            _research_with_limit(
                target=target,
                semaphore=semaphore,
                timeout_seconds=timeout_seconds,
                system_prompt=system_prompt,
                trace_id=trace_id,
                settings=settings,
                stream_writer=writer,
            )
            for target in targets
        ),
        return_exceptions=True,
    )

    successful_count = 0
    fallback_queries = list(state.get("fallback_queries", []))
    for target, item in zip(targets, gathered):
        if isinstance(item, BaseException):
            message = _exception_message(item)
            errors.append(f"Researcher: {target} | {message}")
            logger.error("researcher_node target failed target=%r error=%s", target, message)
            emit(
                {
                    "trace_id": trace_id,
                    "event": "error",
                    "node": "researcher",
                    "payload": {
                        "type": type(item).__name__,
                        "message": message,
                        "question": target[:200],
                    },
                }
            )
            continue

        successful_count += 1
        research_results[item.target] = item.summary
        citations[item.target] = _merge_citations(
            citations.get(item.target, []),
            item.citations,
        )
        if item.fallback_query and item.fallback_query not in fallback_queries:
            fallback_queries.append(item.fallback_query)
        logger.info(
            "researcher_node target done target=%r chars=%s citations=%s",
            target,
            len(item.summary),
            len(item.citations),
        )

    latency_ms = (time.perf_counter() - started_at) * 1000
    emit(
        {
            "trace_id": trace_id,
            "event": "node_end",
            "node": "researcher",
            "payload": {
                "status": "completed" if successful_count else "failed",
                "target_count": len(targets),
                "successful_count": successful_count,
                "research_result_count": len(research_results),
                "error_count": len(errors),
                "revision_mode": revision_mode,
                "latency_ms": latency_ms,
            },
        }
    )
    result: dict[str, object] = {
        "research_results": research_results,
        "citations": citations,
        "errors": errors,
        "fallback_queries": fallback_queries,
        "missing_aspects": [],
        "revision_count": revision_count,
    }
    if not revision_mode and successful_count == 0 and not research_results:
        result["retry_count"] = state.get("retry_count", 0) + 1
    logger.info(
        "researcher_node output successful=%s results=%s errors=%s latency_ms=%.3f",
        successful_count,
        len(research_results),
        len(errors),
        latency_ms,
    )
    return result


async def _research_with_limit(
    *,
    target: str,
    semaphore: asyncio.Semaphore,
    timeout_seconds: float,
    system_prompt: str,
    trace_id: str,
    settings: Settings,
    stream_writer: StreamWriter,
) -> ResearchOutcome:
    async with semaphore:
        return await asyncio.wait_for(
            _research_question(
                question=target,
                system_prompt=system_prompt,
                trace_id=trace_id,
                settings=settings,
                stream_writer=stream_writer,
            ),
            timeout=timeout_seconds,
        )


async def _research_question(
    *,
    question: str,
    system_prompt: str,
    trace_id: str,
    settings: Settings,
    stream_writer: StreamWriter,
) -> ResearchOutcome:
    normalized_question = question.strip()
    if not normalized_question:
        raise ResearcherError("question must not be empty")

    kb_result, kb_error = await _search_kb(normalized_question, trace_id)
    evidence = _kb_evidence(kb_result["hits"])
    citations = _citations_from_evidence(evidence)
    max_score = kb_result["max_score"]
    fallback_query: str | None = None
    if kb_error or max_score < settings.kb_score_threshold:
        fallback_query = normalized_question
        fallback_payload = {
            "node": "researcher",
            "query": normalized_question,
            "max_score": max_score,
            "threshold": settings.kb_score_threshold,
            "reason": "kb_error" if kb_error else "low_score",
        }
        emit(
            {
                "trace_id": trace_id,
                "event": "fallback",
                "node": "researcher",
                "payload": fallback_payload,
            }
        )
        _write_custom_event(stream_writer, "fallback", fallback_payload)
        web_results = await asyncio.to_thread(
            web_search,
            normalized_question,
            max_results=MAX_SEARCH_RESULTS,
        )
        web_evidence = _web_evidence(web_results)
        evidence.extend(web_evidence)
        citations = _merge_citations(
            citations,
            _citations_from_evidence(web_evidence),
        )

    if not evidence:
        raise ResearcherError("no usable local or web evidence")
    model_result = await _call_summary_model(
        question=normalized_question,
        evidence=evidence,
        system_prompt=system_prompt,
        trace_id=trace_id,
    )
    summary = model_result if isinstance(model_result, str) else model_result.content
    return ResearchOutcome(
        target=normalized_question,
        summary=_append_sources(summary, citations),
        citations=citations,
        fallback_query=fallback_query,
    )


async def _search_kb(query: str, trace_id: str) -> tuple[KBSearchResult, str | None]:
    try:
        result = await asyncio.to_thread(
            kb_search,
            query,
            MAX_KB_RESULTS,
            trace_id=trace_id,
        )
        return result, None
    except Exception as exc:
        logger.warning("kb search failed; falling back query=%r error=%s", query, exc)
        return {"hits": [], "max_score": 0.0}, str(exc)


async def _call_summary_model(
    question: str,
    evidence: list[Evidence],
    system_prompt: str,
    trace_id: str,
) -> LLMResult | str:
    try:
        return await achat(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": _build_summary_prompt(question, evidence),
                },
            ],
            node="researcher",
            trace_id=trace_id,
        )
    except LLMError as exc:
        raise ResearcherError(str(exc)) from exc


def _build_summary_prompt(question: str, evidence: list[Evidence]) -> str:
    result_blocks = []
    for index, item in enumerate(evidence, start=1):
        result_blocks.append(
            "\n".join(
                [
                    f"{index}. 来源类型: {item['origin']}",
                    f"来源: {item['source']}",
                    f"内容: {item['snippet']}",
                ]
            )
        )
    return "\n\n".join(
        [
            f"子问题:\n{question}",
            "检索证据:",
            "\n\n".join(result_blocks),
            "请只根据以上证据生成资料摘要，并保留来源。",
        ]
    )


def _kb_evidence(hits: list[KBSearchHit]) -> list[Evidence]:
    return [
        {"source": hit["source"], "snippet": hit["text"], "origin": "kb"}
        for hit in hits
    ]


def _web_evidence(results: list[SearchResult]) -> list[Evidence]:
    return [
        {"source": item["url"], "snippet": item["snippet"], "origin": "web"}
        for item in results
    ]


def _citations_from_evidence(evidence: list[Evidence]) -> list[Citation]:
    return [
        {
            "source": item["source"],
            "origin": item["origin"],
            "snippet": item["snippet"][:500],
        }
        for item in evidence
    ]


def _merge_citations(
    existing: list[Citation],
    additions: list[Citation],
) -> list[Citation]:
    merged = list(existing)
    seen = {(item["source"], item["origin"]) for item in existing}
    for item in additions:
        key = (item["source"], item["origin"])
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _append_sources(summary: str, citations: list[Citation]) -> str:
    if "来源:" in summary:
        return summary
    source_lines = "\n".join(f"- {item['source']}" for item in citations)
    return f"{summary}\n\n来源:\n{source_lines}"


def _research_timeout(settings: Settings) -> float:
    return settings.llm_timeout + settings.embedding_timeout + 30.0


def _exception_message(exc: BaseException) -> str:
    if isinstance(exc, asyncio.TimeoutError):
        return "research item timed out"
    return str(exc) or type(exc).__name__


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


def _write_custom_event(
    writer: StreamWriter,
    event: str,
    payload: dict[str, Any],
) -> None:
    try:
        writer({"event": event, "payload": payload})
    except RuntimeError:
        return


__all__ = ["researcher_node"]
