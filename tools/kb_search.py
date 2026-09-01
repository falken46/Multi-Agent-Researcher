"""Agent 可调用的本地知识库搜索工具。"""

from __future__ import annotations

import logging
from typing import TypedDict

from rag.pipeline import search_with_diagnostics

logger = logging.getLogger(__name__)


class KBSearchHit(TypedDict):
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    ranking_score: float
    score: float
    score_kind: str
    fallback_confidence: float | None
    channel: str


class KBSearchResult(TypedDict):
    hits: list[KBSearchHit]
    max_score: float
    fallback_confidence: float
    fallback_confidence_kind: str


def kb_search(
    query: str,
    top_n: int = 5,
    *,
    trace_id: str | None = None,
) -> KBSearchResult:
    """查询本地知识库，只负责 IO 封装，不决定是否联网降级。"""
    logger.info("kb_search input query=%r top_n=%s", query[:200], top_n)
    diagnostics = search_with_diagnostics(query, top_n=top_n, trace_id=trace_id)
    hits: list[KBSearchHit] = [
        {
            "chunk_id": item.chunk_id,
            "text": item.text,
            "source": item.source,
            "chunk_index": item.chunk_index,
            "ranking_score": item.ranking_score,
            "score": item.score,
            "score_kind": item.score_kind,
            "fallback_confidence": item.fallback_confidence,
            "channel": item.channel,
        }
        for item in diagnostics.hits
    ]
    output: KBSearchResult = {
        "hits": hits,
        "max_score": max((hit["score"] for hit in hits), default=0.0),
        "fallback_confidence": diagnostics.fallback_confidence,
        "fallback_confidence_kind": diagnostics.fallback_confidence_kind,
    }
    logger.info(
        "kb_search output hits=%s ranking_max_score=%.6f "
        "fallback_confidence=%.6f confidence_kind=%s",
        len(output["hits"]),
        output["max_score"],
        output["fallback_confidence"],
        output["fallback_confidence_kind"],
    )
    return output


__all__ = ["KBSearchHit", "KBSearchResult", "kb_search"]
