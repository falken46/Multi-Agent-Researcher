"""jieba 分词与可持久化 BM25 关键词索引。

**关于"增量"能到什么程度（v3 Phase 20）**

`rank_bm25` 的 `BM25Okapi` 在构造时一次性算完 IDF 等全语料统计量，没有提供追加接口。
所以引擎对象**必须重建**，这一步绕不过去。

真正能省掉的是**分词**：jieba 对每篇文档切词是这里最贵的操作，而它的结果只和单篇文档有关，
与语料整体无关。因此本模块把分词结果一起持久化（索引格式 v2），于是：

- :meth:`BM25Index.extend` 只对新增文档分词，已有文档的 token 直接复用
- :meth:`BM25Index._load` 不再重新分词（v1 格式每次加载都要把全语料切一遍）

**诚实的边界：这是"增量分词"，不是"增量 BM25"。** 语料规模再大一个数量级时，
正确的做法是换成 Elasticsearch 之类支持真正增量写入的检索引擎。
"""

from __future__ import annotations

import logging
import pickle
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag.models import Chunk, RetrievalResult

logger = logging.getLogger(__name__)
INDEX_VERSION = 2
_LEGACY_VERSIONS = (1,)  # v1 没有 token 缓存，加载时补算一次并提示升级


class BM25Index:
    """保存原始 chunk，加载时重建轻量 BM25 运行时对象。"""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._chunks: list[Chunk] = []
        self._tokens: list[list[str]] = []
        self._engine: Any | None = None
        if self._path.exists():
            self._load()

    def add(self, chunks: Sequence[Chunk]) -> None:
        """全量替换索引内容。语义与 v2 一致，建全新索引时用。"""
        self._chunks = list(chunks)
        self._tokens = [tokenize(chunk.text) for chunk in self._chunks]
        self._engine = _build_engine(self._tokens)
        self._save()

    def extend(self, chunks: Sequence[Chunk]) -> int:
        """增量追加或更新文档，返回本次实际分词的文档数。

        同 ``id`` 的文档按新内容覆盖（与向量库的 upsert 语义一致）；
        内容完全没变的文档直接跳过，连分词都省掉。
        """
        if not chunks:
            return 0

        position_by_id = {chunk.id: index for index, chunk in enumerate(self._chunks)}
        tokenized = 0
        for chunk in chunks:
            existing = position_by_id.get(chunk.id)
            if existing is not None and self._chunks[existing].text == chunk.text:
                # 内容没变，只可能是 metadata 变了，不用重新分词
                self._chunks[existing] = chunk
                continue
            tokens = tokenize(chunk.text)
            tokenized += 1
            if existing is None:
                position_by_id[chunk.id] = len(self._chunks)
                self._chunks.append(chunk)
                self._tokens.append(tokens)
            else:
                self._chunks[existing] = chunk
                self._tokens[existing] = tokens

        # 引擎必须重建：IDF 依赖全语料统计量，rank_bm25 没有追加接口。
        # 但重建只是从已有 token 重算统计，不再触发 jieba。
        self._engine = _build_engine(self._tokens)
        self._save()
        logger.info(
            "bm25 extend total=%s tokenized=%s reused=%s",
            len(self._chunks),
            tokenized,
            len(chunks) - tokenized,
        )
        return tokenized

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
            # 缓存分词结果：加载与增量追加都不必再对旧文档跑 jieba
            "tokens": self._tokens,
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
        if not isinstance(payload, dict):
            raise RuntimeError(f"unsupported BM25 index format: {self._path}")
        version = payload.get("version")
        if version != INDEX_VERSION and version not in _LEGACY_VERSIONS:
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

        raw_tokens = payload.get("tokens")
        if (
            version == INDEX_VERSION
            and isinstance(raw_tokens, list)
            and len(raw_tokens) == len(self._chunks)
        ):
            self._tokens = [[str(token) for token in row] for row in raw_tokens]
        else:
            # v1 索引没存 token。**不报错**——数据是完整的，只是缺缓存，
            # 补算一次即可；下次 save 时就会写成 v2 格式。
            logger.info(
                "bm25 index %s is legacy v%s, tokenizing %s chunks once to upgrade",
                self._path,
                version,
                len(self._chunks),
            )
            self._tokens = [tokenize(chunk.text) for chunk in self._chunks]

        self._engine = _build_engine(self._tokens)


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


def _build_engine(tokens: Sequence[Sequence[str]]) -> Any | None:
    """从已分好的 token 构造 BM25 引擎。

    这里刻意接收 token 而不是 Chunk —— 调用方负责分词与缓存，
    这样引擎重建就不会隐式触发 jieba。
    """
    if not tokens:
        return None
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise RuntimeError("BM25 index requires the rank-bm25 package") from exc
    return BM25Okapi([list(row) for row in tokens])


__all__ = ["BM25Index", "tokenize"]
