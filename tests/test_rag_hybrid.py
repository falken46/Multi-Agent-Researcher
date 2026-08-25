from __future__ import annotations

import pytest

from rag.hybrid import rrf_fuse
from rag.models import RetrievalResult


def test_rrf_fuses_rankings_and_deduplicates_shared_chunks() -> None:
    vector = [_result("shared", "vector"), _result("vector-only", "vector")]
    bm25 = [_result("bm25-only", "bm25"), _result("shared", "bm25")]

    fused = rrf_fuse({"vector": vector, "bm25": bm25}, rrf_k=60)

    assert [item.chunk_id for item in fused] == [
        "shared",
        "bm25-only",
        "vector-only",
    ]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[0].channel == "bm25+vector"


def _result(chunk_id: str, channel: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=chunk_id,
        source=f"{chunk_id}.md",
        chunk_index=0,
        score=1.0,
        channel=channel,
    )
