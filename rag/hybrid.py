"""基于排名而非原始分数量纲的 RRF 融合。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from rag.models import RetrievalResult


def rrf_fuse(
    channels: Mapping[str, Sequence[RetrievalResult]],
    *,
    rrf_k: int,
) -> list[RetrievalResult]:
    """按 ``sum(1 / (k + rank))`` 融合并去重候选。"""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero")

    scores: dict[str, float] = {}
    results: dict[str, RetrievalResult] = {}
    memberships: dict[str, set[str]] = {}
    for channel_name in sorted(channels):
        for rank, result in enumerate(channels[channel_name], start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (
                rrf_k + rank
            )
            results.setdefault(result.chunk_id, result)
            memberships.setdefault(result.chunk_id, set()).add(channel_name)

    fused = [
        replace(
            result,
            score=scores[chunk_id],
            channel="+".join(sorted(memberships[chunk_id])),
        )
        for chunk_id, result in results.items()
    ]
    return sorted(fused, key=lambda item: (-item.score, item.chunk_id))


__all__ = ["rrf_fuse"]
