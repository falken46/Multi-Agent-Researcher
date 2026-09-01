"""jieba 分词与可持久化 BM25 关键词索引。"""

from __future__ import annotations

import logging
import pickle
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag.models import Chunk, RetrievalResult

logger = logging.getLogger(__name__)
INDEX_VERSION = 1


class BM25Index:
    """保存原始 chunk，加载时重建轻量 BM25 运行时对象。"""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._chunks: list[Chunk] = []
        self._engine: Any | None = None
        if self._path.exists():
            self._load()

    def add(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._engine = _create_engine(self._chunks)
        self._save()

    def query(self, query: str, *, top_k: int) -> list[RetrievalResult]:
        if top_k <= 0 or self._engine is None or not self._chunks:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._engine.get_scores(tokens)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: (-float(item[1]), self._chunks[item[0]].id),
        )
        results = []
        for index, raw_score in ranked:
            score = float(raw_score)
            if score <= 0:
                continue
            chunk = self._chunks[index]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.id,
                    text=chunk.text,
                    source=str(chunk.metadata.get("source_path", "unknown")),
                    chunk_index=int(chunk.metadata.get("chunk_index", 0)),
                    score=score,
                    channel="bm25",
                    metadata=dict(chunk.metadata),
                    fallback_confidence=None,
                    score_kind="bm25",
                )
            )
            if len(results) >= top_k:
                break
        return results

    def count(self) -> int:
        return len(self._chunks)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_VERSION,
            "chunks": [
                {"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata}
                for chunk in self._chunks
            ],
        }
        temporary_path = self._path.with_suffix(self._path.suffix + ".tmp")
        with temporary_path.open("wb") as index_file:
            pickle.dump(payload, index_file)
        temporary_path.replace(self._path)

    def _load(self) -> None:
        try:
            with self._path.open("rb") as index_file:
                payload = pickle.load(index_file)
        except (OSError, pickle.UnpicklingError, EOFError) as exc:
            raise RuntimeError(f"failed to load BM25 index {self._path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("version") != INDEX_VERSION:
            raise RuntimeError(f"unsupported BM25 index format: {self._path}")
        raw_chunks = payload.get("chunks")
        if not isinstance(raw_chunks, list):
            raise RuntimeError(f"BM25 index is missing chunks: {self._path}")
        self._chunks = [
            Chunk(
                id=str(item["id"]),
                text=str(item["text"]),
                metadata=dict(item.get("metadata", {})),
            )
            for item in raw_chunks
        ]
        self._engine = _create_engine(self._chunks)


def tokenize(text: str) -> list[str]:
    """文档与查询共用的 jieba 精确模式分词和规范化。"""
    try:
        import jieba
    except ImportError as exc:
        raise RuntimeError("BM25 index requires the jieba package") from exc
    jieba.setLogLevel(logging.WARNING)
    return [
        token.lower()
        for token in jieba.lcut(text, cut_all=False)
        if re.search(r"[\w\u4e00-\u9fff]", token, flags=re.UNICODE)
    ]


def _create_engine(chunks: Sequence[Chunk]) -> Any | None:
    if not chunks:
        return None
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise RuntimeError("BM25 index requires the rank-bm25 package") from exc
    return BM25Okapi([tokenize(chunk.text) for chunk in chunks])


__all__ = ["BM25Index", "tokenize"]
