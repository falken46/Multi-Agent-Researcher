"""知识库建库与混合检索的统一入口。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from core.config import Settings, get_settings
from core.trace import emit
from rag.bm25 import BM25Index
from rag.embeddings import create_embedding_backend
from rag.hybrid import rrf_fuse
from rag.loader import LoadFailure, load_directory
from rag.models import RetrievalResult
from rag.rerank import rerank
from rag.splitter import split_documents
from rag.vectorstore import ChromaVectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildReport:
    source_dir: str
    document_count: int
    chunk_count: int
    vector_count: int
    bm25_count: int
    load_failures: list[LoadFailure]
    channel_errors: dict[str, str]


def build_index(
    directory: Path | str | None = None,
    *,
    settings: Settings | None = None,
) -> BuildReport:
    """加载、切分并分别构建 Chroma 与 BM25 索引。"""
    current = settings or get_settings()
    source_dir = Path(directory) if directory is not None else current.kb_dir
    logger.info("kb build input directory=%s", source_dir)
    load_report = load_directory(source_dir)
    chunks = split_documents(
        load_report.documents,
        chunk_size=current.chunk_size,
        chunk_overlap=current.chunk_overlap,
    )
    if not chunks:
        raise RuntimeError(f"no usable chunks found in {source_dir}")

    vector_count = 0
    bm25_count = 0
    channel_errors: dict[str, str] = {}
    if current.vector_search_enabled:
        try:
            embedding_backend = create_embedding_backend(current)
            embeddings = embedding_backend.embed_documents(
                [chunk.text for chunk in chunks]
            )
            vector_store = ChromaVectorStore(
                current.chroma_dir,
                collection_name=current.chroma_collection,
                reset=True,
            )
            vector_store.add(chunks, embeddings)
            vector_count = vector_store.count()
        except Exception as exc:
            channel_errors["vector"] = str(exc)
            logger.exception("kb vector build failed")

    if current.bm25_search_enabled:
        try:
            bm25_index = BM25Index(current.bm25_index_path)
            bm25_index.add(chunks)
            bm25_count = bm25_index.count()
        except Exception as exc:
            channel_errors["bm25"] = str(exc)
            logger.exception("kb BM25 build failed")

    if vector_count == 0 and bm25_count == 0:
        detail = "; ".join(
            f"{channel}: {message}" for channel, message in channel_errors.items()
        )
        raise RuntimeError(f"all enabled index channels failed: {detail or 'none enabled'}")
    logger.info(
        "kb build output documents=%s chunks=%s vector=%s bm25=%s failures=%s",
        len(load_report.documents),
        len(chunks),
        vector_count,
        bm25_count,
        len(load_report.failures),
    )
    return BuildReport(
        source_dir=str(source_dir),
        document_count=len(load_report.documents),
        chunk_count=len(chunks),
        vector_count=vector_count,
        bm25_count=bm25_count,
        load_failures=load_report.failures,
        channel_errors=channel_errors,
    )


def search(
    query: str,
    top_n: int | None = None,
    *,
    trace_id: str | None = None,
    settings: Settings | None = None,
) -> list[RetrievalResult]:
    """查询可用通道、RRF 融合并按配置重排。"""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty")
    current = settings or get_settings()
    result_limit = top_n or current.rerank_top_n
    if result_limit <= 0:
        raise ValueError("top_n must be greater than zero")
    started_at = time.perf_counter()
    logger.info("kb search input query=%r top_n=%s", normalized_query[:200], result_limit)

    channel_results: dict[str, list[RetrievalResult]] = {}
    channel_errors: dict[str, str] = {}
    if current.vector_search_enabled:
        try:
            embedding_backend = create_embedding_backend(current)
            vector_store = ChromaVectorStore(
                current.chroma_dir,
                collection_name=current.chroma_collection,
            )
            channel_results["vector"] = vector_store.query(
                embedding_backend.embed_query(normalized_query),
                top_k=current.retrieval_top_k,
            )
        except Exception as exc:
            channel_errors["vector"] = str(exc)
            logger.warning("kb vector search degraded error=%s", exc)

    if current.bm25_search_enabled:
        try:
            bm25_index = BM25Index(current.bm25_index_path)
            channel_results["bm25"] = bm25_index.query(
                normalized_query,
                top_k=current.retrieval_top_k,
            )
        except Exception as exc:
            channel_errors["bm25"] = str(exc)
            logger.warning("kb BM25 search degraded error=%s", exc)

    available_results = {
        channel: results for channel, results in channel_results.items() if results
    }
    fused = rrf_fuse(available_results, rrf_k=current.rrf_k)
    try:
        results = rerank(
            normalized_query,
            fused,
            top_n=result_limit,
            trace_id=trace_id,
            settings=current,
        )
    except Exception as exc:
        channel_errors["rerank"] = str(exc)
        logger.warning("kb rerank degraded error=%s", exc)
        results = fused[:result_limit]

    latency_ms = (time.perf_counter() - started_at) * 1000
    logger.info(
        "kb search output hits=%s channels=%s errors=%s latency_ms=%.3f",
        len(results),
        sorted(available_results),
        sorted(channel_errors),
        latency_ms,
    )
    if trace_id:
        emit(
            {
                "trace_id": trace_id,
                "event": "retrieval",
                "node": "kb_search",
                "payload": {
                    "query": normalized_query[:200],
                    "hit_count": len(results),
                    "channels": sorted(available_results),
                    "channel_errors": channel_errors,
                    "max_score": max((item.score for item in results), default=0.0),
                    "latency_ms": latency_ms,
                },
            }
        )
    return results


__all__ = ["BuildReport", "build_index", "search"]
