"""向量库的窄接口封装。

对检索流水线只暴露 ``add`` / ``query`` / ``count`` 三个方法，供应商细节全部关在
适配器内部。目前有两个实现：

- :class:`ChromaVectorStore`  —— 嵌入式，零运维，v2 使用
- :class:`MilvusVectorStore`  —— 独立服务或 Milvus Lite，v3 使用

由 :func:`create_vector_store` 按 ``VECTOR_BACKEND`` 配置选择。保留 Chroma 是为了
Phase 17 能做 A/B 对照：换库之后 R 轨结果必须与 Chroma 版基本一致，不一致说明
适配器有 bug，而不是换库本身的效果。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag.embeddings import Embedding
from rag.models import Chunk, Metadata, RetrievalResult

if TYPE_CHECKING:  # 仅供类型检查，运行时不导入，避免 core.config 与 rag 循环依赖
    from core.config import Settings


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


class MilvusVectorStore:
    """Milvus 的窄接口封装，方法签名与 :class:`ChromaVectorStore` 完全一致。

    ``uri`` 同时支持两种形态，由 pymilvus 自行判别：

    - ``http://localhost:19530`` —— 独立 Milvus 服务（生产 / compose）
    - ``./data/milvus_lite.db``  —— Milvus Lite 本地文件（测试 / CI，需装 ``milvus-lite``）

    与 Chroma 的两处**实质差异**（不是风格差异，抄错会出静默的错误结果）：

    1. **距离语义相反。** Chroma 的 ``distance`` 是余弦*距离*（越小越相似，
       ``similarity = 1 - distance``）；Milvus 在 ``metric_type="COSINE"`` 下返回的
       ``distance`` **就是余弦相似度本身**（越大越相似，相同向量为 1.0、正交为 0.0）。
       照抄 Chroma 的换算会把排序整个倒过来，而且不会报错。
    2. **集合必须先有维度才能写入。** Chroma 可以先建集合再随便写；Milvus 建集合时就要
       声明 ``dim``。所以这里推迟到第一次 ``add`` 时按 embedding 长度建集合。
    """

    _ID_MAX_LENGTH = 512
    _TEXT_MAX_LENGTH = 65535

    def __init__(
        self,
        uri: str,
        *,
        collection_name: str,
        token: str = "",
        reset: bool = False,
    ) -> None:
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise RuntimeError("vector store requires the pymilvus package") from exc

        normalized_uri = str(uri).strip()
        if not normalized_uri:
            raise ValueError("MILVUS_URI must not be empty")
        # 本地文件形态要保证父目录存在，否则 milvus-lite 直接报文件打不开
        if not normalized_uri.startswith(("http://", "https://", "unix:")):
            Path(normalized_uri).parent.mkdir(parents=True, exist_ok=True)

        self._client = MilvusClient(uri=normalized_uri, token=token)
        self._collection_name = collection_name
        self._loaded = False
        if reset and self._client.has_collection(collection_name):
            self._client.drop_collection(collection_name)

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
        """用稳定 chunk id 幂等写入向量与元数据。"""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        self._ensure_collection(dim=len(embeddings[0]))
        rows = [
            {
                "id": chunk.id,
                "vector": [float(value) for value in vector],
                "text": chunk.text,
                "metadata": _sanitize_metadata(chunk.metadata),
            }
            for chunk, vector in zip(chunks, embeddings)
        ]
        # upsert 而非 insert：同一个 chunk id 重复建索引时覆盖，保持幂等
        self._client.upsert(collection_name=self._collection_name, data=rows)

    def query(self, embedding: Embedding, *, top_k: int) -> list[RetrievalResult]:
        """按余弦相似度查询并转换为统一结果结构。"""
        if top_k <= 0 or self.count() == 0:
            return []
        self._ensure_loaded()
        payload = self._client.search(
            collection_name=self._collection_name,
            data=[list(embedding)],
            limit=top_k,
            output_fields=["text", "metadata"],
            search_params={"metric_type": "COSINE"},
        )
        hits = payload[0] if payload else []
        results = []
        for hit in hits:
            entity = hit.get("entity") or {}
            metadata = _sanitize_metadata(entity.get("metadata") or {})
            # ⚠️ Milvus COSINE 的 distance 就是相似度本身，不要写成 1 - distance
            similarity = max(0.0, min(1.0, float(hit.get("distance", 0.0))))
            results.append(
                RetrievalResult(
                    chunk_id=str(hit.get("id", "")),
                    text=str(entity.get("text") or ""),
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
        if not self._client.has_collection(self._collection_name):
            return 0
        self._ensure_loaded()
        # 用 count(*) 而不是 get_collection_stats：后者在真实 Milvus 上是近似值，
        # 插入后未 flush 时会滞后，用来做 "空集合就短路" 的判断不可靠。
        rows = self._client.query(
            collection_name=self._collection_name,
            filter="",
            output_fields=["count(*)"],
        )
        if not rows:
            return 0
        return int(dict(rows[0]).get("count(*)", 0))

    def _ensure_loaded(self) -> None:
        """搜索前把集合载入内存。

        这是 Milvus 相对 Chroma 多出来的一步，也是本次迁移踩到的实际坑：
        集合建好写完之后，**换一个进程或换一个 client 实例再来查，它是 ``released``
        状态**，直接 search 会报 ``code=101 ... call load() before search``。
        建集合的那个 client 因为刚建完是隐式加载的，所以单元测试里不复现 ——
        只有"建索引"和"查询"分处两次运行时才会暴露（正是 pipeline 的真实用法）。

        ``load_collection`` 幂等，已加载时是廉价调用，所以这里每个实例只挡一次即可。
        """
        if self._loaded:
            return
        self._client.load_collection(self._collection_name)
        self._loaded = True

    def _ensure_collection(self, *, dim: int) -> None:
        """首次写入时按 embedding 维度建集合；已存在则直接复用。"""
        if self._client.has_collection(self._collection_name):
            return
        from pymilvus import DataType

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            "id", DataType.VARCHAR, is_primary=True, max_length=self._ID_MAX_LENGTH
        )
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("text", DataType.VARCHAR, max_length=self._TEXT_MAX_LENGTH)
        schema.add_field("metadata", DataType.JSON)

        index_params = self._client.prepare_index_params()
        # FLAT = 暴力精确检索。本项目语料只有百余条，精确检索又快又没有召回损失；
        # 上万条以上再换 HNSW（近似检索，用少量召回率换速度）。
        index_params.add_index(
            field_name="vector", index_type="FLAT", metric_type="COSINE"
        )
        self._client.create_collection(
            collection_name=self._collection_name,
            schema=schema,
            index_params=index_params,
        )


# 上层只依赖这个别名，不依赖具体供应商类型
VectorStore = ChromaVectorStore | MilvusVectorStore


def create_vector_store(
    settings: Settings,
    *,
    reset: bool = False,
    collection_name: str | None = None,
    chroma_dir: Path | None = None,
) -> VectorStore:
    """按 ``VECTOR_BACKEND`` 选择实现。

    保留 chroma 分支是刻意的：Phase 17 要拿两个后端跑同一套 R 轨做对照，
    结果不一致说明 Milvus 适配器有 bug。对照做完之前不要删。

    ``collection_name`` / ``chroma_dir`` 是给评测轨用的覆盖项 —— 评测要建独立索引，
    不能污染开发用的知识库集合。
    """
    backend = settings.vector_backend
    if backend == "chroma":
        return ChromaVectorStore(
            chroma_dir if chroma_dir is not None else settings.chroma_dir,
            collection_name=collection_name or settings.chroma_collection,
            reset=reset,
        )
    if backend == "milvus":
        return MilvusVectorStore(
            settings.milvus_uri,
            collection_name=collection_name or settings.milvus_collection,
            token=settings.milvus_token.get_secret_value(),
            reset=reset,
        )
    raise ValueError(f"unsupported VECTOR_BACKEND: {backend!r}")


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


__all__ = [
    "ChromaVectorStore",
    "MilvusVectorStore",
    "VectorStore",
    "create_vector_store",
]
