from __future__ import annotations

from pathlib import Path

import pytest

from core.config import Settings, clear_settings_cache, get_settings
from rag.pipeline import build_index, search
from tools.kb_search import kb_search


def test_hybrid_pipeline_builds_searches_and_degrades_by_channel(
    runtime_dir: Path,
) -> None:
    settings = _settings(runtime_dir)

    report = build_index(Path("data/kb"), settings=settings)
    relevant = search("RRF 为什么适合混合检索", settings=settings)
    unrelated = search("量子纠缠烹饪配方", settings=settings)

    assert report.document_count == 20
    assert report.chunk_count >= 20
    assert report.vector_count == report.chunk_count
    assert report.bm25_count == report.chunk_count
    assert relevant[0].source == "13_rrf.md"
    assert relevant[0].score >= unrelated[0].score * 1.8

    bm25_only = search(
        "RRF 混合检索",
        settings=settings.model_copy(update={"vector_search_enabled": False}),
    )
    vector_only = search(
        "语义召回和关键词排名如何融合",
        settings=settings.model_copy(update={"bm25_search_enabled": False}),
    )

    assert bm25_only and bm25_only[0].channel == "bm25"
    assert vector_only and vector_only[0].channel == "vector"


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
    assert result["hits"][0]["source"] == "12_bm25.md"
    assert result["max_score"] == max(hit["score"] for hit in result["hits"])


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
