"""Chroma 持久化向量库的窄接口封装。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag.embeddings import Embedding
from rag.models import Chunk, Metadata, RetrievalResult


class ChromaVectorStore:
    """仅向检索流水线暴露 add、query 与 count。"""

    def __init__(
        self,
        path: Path,
        *,
        collection_name: str,
        reset: bool = False,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("vector store requires the chromadb package") from exc

        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        if reset and collection_name in _collection_names(self._client):
            self._client.delete_collection(collection_name)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
        """用稳定 chunk id 幂等写入向量与元数据。"""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[_sanitize_metadata(chunk.metadata) for chunk in chunks],
            embeddings=[list(vector) for vector in embeddings],
        )

    def query(self, embedding: Embedding, *, top_k: int) -> list[RetrievalResult]:
        """按余弦距离查询并转换为统一结果结构。"""
        if top_k <= 0 or self.count() == 0:
            return []
        payload = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        ids = _first_list(payload.get("ids"))
        documents = _first_list(payload.get("documents"))
        metadatas = _first_list(payload.get("metadatas"))
        distances = _first_list(payload.get("distances"))
        results = []
        for chunk_id, text, raw_metadata, raw_distance in zip(
            ids, documents, metadatas, distances
        ):
            metadata = _sanitize_metadata(raw_metadata or {})
            distance = float(raw_distance)
            similarity = max(0.0, min(1.0, 1.0 - distance))
            results.append(
                RetrievalResult(
                    chunk_id=str(chunk_id),
                    text=str(text or ""),
                    source=str(metadata.get("source_path", "unknown")),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    score=similarity,
                    channel="vector",
                    metadata=metadata,
                    fallback_confidence=similarity,
                    score_kind="cosine_similarity",
                )
            )
        return results

    def count(self) -> int:
        return int(self._collection.count())


def _collection_names(client: Any) -> set[str]:
    names = set()
    for collection in client.list_collections():
        names.add(str(getattr(collection, "name", collection)))
    return names


def _sanitize_metadata(metadata: dict[str, Any]) -> Metadata:
    sanitized: Metadata = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            sanitized[str(key)] = value
        elif value is not None:
            sanitized[str(key)] = str(value)
    return sanitized


def _first_list(value: Any) -> list[Any]:
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    return []


__all__ = ["ChromaVectorStore"]
