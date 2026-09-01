from __future__ import annotations

from pathlib import Path

import pytest

from core.config import Settings, clear_settings_cache, get_settings
from rag.pipeline import build_index, search, search_with_diagnostics
from tools.kb_search import kb_search


def test_hybrid_pipeline_builds_searches_and_degrades_by_channel(
    runtime_dir: Path,
) -> None:
    settings = _settings(runtime_dir)

    report = build_index(Path("data/kb"), settings=settings)
    relevant_diagnostics = search_with_diagnostics(
        "RRF 为什么适合混合检索",
        settings=settings,
    )
    unrelated_diagnostics = search_with_diagnostics(
        "量子纠缠烹饪配方",
        settings=settings,
    )
    relevant = relevant_diagnostics.hits

    assert report.document_count >= 20
    assert report.chunk_count >= report.document_count
    assert report.vector_count == report.chunk_count
    assert report.bm25_count == report.chunk_count
    assert relevant[0].source == "13_rrf.md"
    assert relevant_diagnostics.fallback_confidence >= (
        unrelated_diagnostics.fallback_confidence * 1.8
    )

    bm25_only = search(
        "RRF 混合检索",
        settings=settings.model_copy(update={"vector_search_enabled": False}),
    )
    vector_only = search(
        "语义召回和关键词排名如何融合",
        settings=settings.model_copy(update={"bm25_search_enabled": False}),
    )
    hybrid_diagnostics = search_with_diagnostics(
        "语义召回和关键词排名如何融合",
        settings=settings,
    )
    vector_diagnostics = search_with_diagnostics(
        "语义召回和关键词排名如何融合",
        settings=settings.model_copy(update={"bm25_search_enabled": False}),
    )

    assert bm25_only and bm25_only[0].channel == "bm25"
    assert vector_only and vector_only[0].channel == "vector"
    assert hybrid_diagnostics.hits[0].score_kind == "rrf"
    assert hybrid_diagnostics.fallback_confidence == pytest.approx(
        vector_diagnostics.fallback_confidence
    )
    assert hybrid_diagnostics.fallback_confidence_kind == "vector_cosine_similarity"


def test_kb_search_tool_returns_hits_and_max_score(
    runtime_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_BACKEND", "fake")
    monkeypatch.setenv("CHROMA_DIR", str(runtime_dir / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION", "tool_test")
    monkeypatch.setenv("BM25_INDEX_PATH", str(runtime_dir / "bm25" / "index.pkl"))
    monkeypatch.setenv("RERANK_BACKEND", "none")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "10")
    clear_settings_cache()

    build_index(Path("data/kb"), settings=get_settings())
    result = kb_search("BM25 为什么需要 jieba 中文分词", top_n=3)

    assert result["hits"]
    assert "bm25" in result["hits"][0]["source"].lower()
    assert result["max_score"] == max(hit["score"] for hit in result["hits"])
    assert result["fallback_confidence_kind"] == "vector_cosine_similarity"
    assert 0.0 <= result["fallback_confidence"] <= 1.0
    assert all(hit["ranking_score"] == hit["score"] for hit in result["hits"])


def _settings(runtime_dir: Path) -> Settings:
    return Settings(
        _env_file=None,
        embedding_backend="fake",
        chroma_dir=runtime_dir / "chroma",
        chroma_collection="pipeline_test",
        bm25_index_path=runtime_dir / "bm25" / "index.pkl",
        rerank_backend="none",
        retrieval_top_k=10,
    )
