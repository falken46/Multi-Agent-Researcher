from __future__ import annotations

import math

from rag.embeddings import FakeEmbeddingBackend


def test_fake_embedding_is_deterministic_and_semantically_useful() -> None:
    backend = FakeEmbeddingBackend(dimension=64)

    query = backend.embed_query("RRF 混合检索")
    same_topic = backend.embed_query("RRF 用于混合检索和排名融合")
    unrelated = backend.embed_query("量子纠缠烹饪配方")

    assert query == backend.embed_query("RRF 混合检索")
    assert math.isclose(_dot(query, query), 1.0)
    assert _dot(query, same_topic) > _dot(query, unrelated)


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
