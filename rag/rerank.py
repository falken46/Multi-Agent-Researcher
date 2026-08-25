"""可切换的 ONNX cross-encoder 与 LLM 重排。"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import replace
from functools import lru_cache
from typing import Any

from core.config import Settings, get_settings
from core.llm import chat
from core.prompts import load_prompt
from core.trace import new_trace_id
from rag.models import RetrievalResult

logger = logging.getLogger(__name__)


def rerank(
    query: str,
    candidates: Sequence[RetrievalResult],
    *,
    top_n: int,
    trace_id: str | None = None,
    settings: Settings | None = None,
) -> list[RetrievalResult]:
    """按配置重排；解析或推理失败由调用方决定是否降级。"""
    current = settings or get_settings()
    selected = list(candidates)
    if top_n <= 0 or not selected:
        return []
    if current.rerank_backend == "none":
        return selected[:top_n]
    if current.rerank_backend == "onnx":
        return _onnx_rerank(
            query,
            selected,
            top_n=top_n,
            model_name=current.rerank_model,
        )
    return _llm_rerank(
        query,
        selected,
        top_n=top_n,
        trace_id=trace_id or new_trace_id(),
    )


def _onnx_rerank(
    query: str,
    candidates: list[RetrievalResult],
    *,
    top_n: int,
    model_name: str,
) -> list[RetrievalResult]:
    model = _load_onnx_model(model_name)
    scores = list(model.rerank(query, [item.text for item in candidates]))
    if len(scores) != len(candidates):
        raise RuntimeError("ONNX reranker returned an invalid score count")
    reranked = [
        replace(
            item,
            score=float(score),
            channel=f"{item.channel}+onnx_rerank",
            score_kind="onnx_rerank",
        )
        for item, score in zip(candidates, scores)
    ]
    return sorted(reranked, key=lambda item: (-item.score, item.chunk_id))[:top_n]


@lru_cache(maxsize=4)
def _load_onnx_model(model_name: str) -> Any:
    """按模型名缓存 ONNX 会话，避免评测时为每道 query 重复加载模型。"""

    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError as exc:
        raise RuntimeError("ONNX rerank requires the fastembed package") from exc
    return TextCrossEncoder(model_name=model_name)


def _llm_rerank(
    query: str,
    candidates: list[RetrievalResult],
    *,
    top_n: int,
    trace_id: str,
) -> list[RetrievalResult]:
    payload = {
        "query": query,
        "candidates": [
            {"chunk_id": item.chunk_id, "text": item.text} for item in candidates
        ],
    }
    result = chat(
        [
            {"role": "system", "content": load_prompt("reranker_system")},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        node="reranker",
        trace_id=trace_id,
        json_mode=True,
    )
    try:
        parsed = json.loads(_strip_json_fence(result.content))
        raw_scores = parsed.get("scores", [])
        score_by_id = {
            str(item["chunk_id"]): max(0.0, min(1.0, float(item["score"])))
            for item in raw_scores
            if isinstance(item, dict)
        }
    except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        logger.warning("LLM rerank output could not be parsed; using neutral scores")
        score_by_id = {}

    reranked = [
        replace(
            item,
            score=score_by_id.get(item.chunk_id, 0.5),
            channel=f"{item.channel}+llm_rerank",
            score_kind="llm_rerank",
        )
        for item in candidates
    ]
    return sorted(reranked, key=lambda item: (-item.score, item.chunk_id))[:top_n]


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


__all__ = ["rerank"]
