"""Agent 可调用的本地知识库搜索工具。"""

from __future__ import annotations

import logging
from typing import TypedDict

from rag.pipeline import search

logger = logging.getLogger(__name__)


class KBSearchHit(TypedDict):
    text: str
    source: str
    chunk_index: int
    score: float
    channel: str


class KBSearchResult(TypedDict):
    hits: list[KBSearchHit]
    max_score: float


def kb_search(
    query: str,
    top_n: int = 5,
    *,
    trace_id: str | None = None,
) -> KBSearchResult:
    """查询本地知识库，只负责 IO 封装，不决定是否联网降级。"""
    logger.info("kb_search input query=%r top_n=%s", query[:200], top_n)
    results = search(query, top_n=top_n, trace_id=trace_id)
    hits: list[KBSearchHit] = [
        {
            "text": item.text,
            "source": item.source,
            "chunk_index": item.chunk_index,
            "score": item.score,
            "channel": item.channel,
        }
        for item in results
    ]
    output: KBSearchResult = {
        "hits": hits,
        "max_score": max((hit["score"] for hit in hits), default=0.0),
    }
    logger.info(
        "kb_search output hits=%s max_score=%.6f",
        len(output["hits"]),
        output["max_score"],
    )
    return output


__all__ = ["KBSearchHit", "KBSearchResult", "kb_search"]
