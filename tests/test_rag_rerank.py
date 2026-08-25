from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastembed.rerank import cross_encoder

import rag.rerank as rerank_module
from core.config import Settings
from rag.models import RetrievalResult
from rag.rerank import rerank


def test_onnx_rerank_uses_cross_encoder_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rerank_module._load_onnx_model.cache_clear()
    initialization_count = 0

    class FakeCrossEncoder:
        def __init__(self, model_name: str) -> None:
            nonlocal initialization_count
            assert model_name == "fake-reranker"
            initialization_count += 1

        def rerank(self, query: str, documents: list[str]) -> Iterator[float]:
            assert query == "哪个更相关"
            assert documents == ["候选 A", "候选 B"]
            return iter([0.1, 0.9])

    monkeypatch.setattr(cross_encoder, "TextCrossEncoder", FakeCrossEncoder)
    settings = Settings(
        _env_file=None,
        rerank_backend="onnx",
        rerank_model="fake-reranker",
    )

    results = rerank("哪个更相关", _candidates(), top_n=1, settings=settings)
    rerank("哪个更相关", _candidates(), top_n=1, settings=settings)

    assert results[0].chunk_id == "b"
    assert results[0].score == pytest.approx(0.9)
    assert results[0].channel.endswith("onnx_rerank")
    assert results[0].score_kind == "onnx_rerank"
    assert results[0].fallback_confidence == pytest.approx(0.8)
    assert initialization_count == 1
    rerank_module._load_onnx_model.cache_clear()


def test_llm_rerank_parsing_failure_degrades_to_neutral_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rerank_module,
        "chat",
        lambda *args, **kwargs: SimpleNamespace(content="not-json"),
    )
    settings = Settings(_env_file=None, rerank_backend="llm")

    results = rerank(
        "哪个更相关",
        _candidates(),
        top_n=2,
        trace_id="rerank-test",
        settings=settings,
    )

    assert [item.chunk_id for item in results] == ["a", "b"]
    assert all(item.score == 0.5 for item in results)
    assert all(item.channel.endswith("llm_rerank") for item in results)
    assert all(item.score_kind == "llm_rerank" for item in results)
    assert [item.fallback_confidence for item in results] == [0.7, 0.8]


def _candidates() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            "a",
            "候选 A",
            "a.md",
            0,
            0.03,
            "bm25+vector",
            fallback_confidence=0.7,
            score_kind="rrf",
        ),
        RetrievalResult(
            "b",
            "候选 B",
            "b.md",
            0,
            0.02,
            "vector",
            fallback_confidence=0.8,
            score_kind="rrf",
        ),
    ]
