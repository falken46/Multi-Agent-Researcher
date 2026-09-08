"""向量库适配器测试（Phase 17）。

重点不是测 Milvus 本身（那是上游的事），而是测三件事：

1. Milvus 适配器与 Chroma 适配器**行为一致** —— 这是换库能否成立的前提
2. 余弦语义没有搞反 —— Milvus 的 ``distance`` 是相似度，Chroma 的是距离，
   这是本次迁移最容易出的静默错误
3. 工厂按 ``VECTOR_BACKEND`` 正确分派
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import Settings
from rag.models import Chunk
from rag.vectorstore import (
    ChromaVectorStore,
    MilvusVectorStore,
    create_vector_store,
)

# 四维单位向量，正交关系一眼可见，便于断言排序
_CHUNKS = [
    Chunk(id="c1", text="第一条", metadata={"source_path": "a.md", "chunk_index": 0}),
    Chunk(id="c2", text="第二条", metadata={"source_path": "b.md", "chunk_index": 1}),
    Chunk(id="c3", text="第三条", metadata={"source_path": "c.md", "chunk_index": 2}),
]
_EMBEDDINGS = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.9, 0.1, 0.0, 0.0],  # 与 c1 很接近，用来验证排序而不只是命中
]


def _milvus(runtime_dir: Path, *, reset: bool = False) -> MilvusVectorStore:
    return MilvusVectorStore(
        str(runtime_dir / "milvus_lite.db"),
        collection_name="vectorstore_test",
        reset=reset,
    )


def _chroma(runtime_dir: Path, *, reset: bool = False) -> ChromaVectorStore:
    return ChromaVectorStore(
        runtime_dir / "chroma",
        collection_name="vectorstore_test",
        reset=reset,
    )


def test_milvus_add_query_count_roundtrip(runtime_dir: Path) -> None:
    store = _milvus(runtime_dir)
    assert store.count() == 0
    assert store.query([1.0, 0.0, 0.0, 0.0], top_k=3) == []

    store.add(_CHUNKS, _EMBEDDINGS)
    assert store.count() == 3

    hits = store.query([1.0, 0.0, 0.0, 0.0], top_k=3)
    assert [hit.chunk_id for hit in hits] == ["c1", "c3", "c2"]
    assert hits[0].text == "第一条"
    assert hits[0].source == "a.md"
    assert hits[0].chunk_index == 0
    assert hits[0].channel == "vector"
    assert hits[0].score_kind == "cosine_similarity"


def test_milvus_cosine_semantics_not_inverted(runtime_dir: Path) -> None:
    """Milvus 的 distance 就是相似度，写成 1 - distance 会让这个测试失败。

    这是本次迁移的核心回归点：抄错不会报错，只会静默地把排序倒过来。
    """
    store = _milvus(runtime_dir)
    store.add(_CHUNKS, _EMBEDDINGS)

    hits = store.query([1.0, 0.0, 0.0, 0.0], top_k=3)
    scores = {hit.chunk_id: hit.score for hit in hits}

    # 完全相同的向量必须接近 1.0，正交向量必须接近 0.0
    assert scores["c1"] == pytest.approx(1.0, abs=1e-3)
    assert scores["c2"] == pytest.approx(0.0, abs=1e-3)
    # 相近向量排在正交向量前面
    assert scores["c3"] > scores["c2"]
    # 结果本身按分数降序
    assert [hit.score for hit in hits] == sorted(
        (hit.score for hit in hits), reverse=True
    )


def test_milvus_matches_chroma_ranking(runtime_dir: Path) -> None:
    """同一批数据，两个后端必须给出相同的排序与相近的分数。

    这是 Phase 17 验收标准的最小版本：换库不改变检索结果。
    """
    milvus_store = _milvus(runtime_dir)
    chroma_store = _chroma(runtime_dir)
    milvus_store.add(_CHUNKS, _EMBEDDINGS)
    chroma_store.add(_CHUNKS, _EMBEDDINGS)

    query = [1.0, 0.0, 0.0, 0.0]
    milvus_results = milvus_store.query(query, top_k=3)
    chroma_results = chroma_store.query(query, top_k=3)

    assert [hit.chunk_id for hit in milvus_results] == [
        hit.chunk_id for hit in chroma_results
    ]
    for m_hit, c_hit in zip(milvus_results, chroma_results):
        assert m_hit.score == pytest.approx(c_hit.score, abs=1e-3)
        assert m_hit.text == c_hit.text
        assert m_hit.source == c_hit.source
        assert m_hit.chunk_index == c_hit.chunk_index


def test_milvus_query_from_a_fresh_instance_after_build(runtime_dir: Path) -> None:
    """建索引与查询分处两个 client 实例时仍然可查。

    这是 Phase 17 实际踩到的坑：Milvus 的集合对新连接是 ``released`` 状态，
    不显式 ``load_collection`` 就报 ``code=101``。建集合的那个实例是隐式加载的，
    所以"建完立刻查"的测试抓不到 —— 必须换一个实例查，才复现 pipeline 的真实用法
    （``build_index`` 和 ``search`` 是两次独立构造）。
    """
    builder = _milvus(runtime_dir, reset=True)
    builder.add(_CHUNKS, _EMBEDDINGS)

    reader = _milvus(runtime_dir)  # 全新实例，模拟另一次运行
    assert reader.count() == 3
    hits = reader.query([1.0, 0.0, 0.0, 0.0], top_k=3)
    assert [hit.chunk_id for hit in hits] == ["c1", "c3", "c2"]


def test_milvus_upsert_is_idempotent(runtime_dir: Path) -> None:
    store = _milvus(runtime_dir)
    store.add(_CHUNKS, _EMBEDDINGS)
    store.add(_CHUNKS, _EMBEDDINGS)
    assert store.count() == 3


def test_milvus_reset_drops_existing_collection(runtime_dir: Path) -> None:
    _milvus(runtime_dir).add(_CHUNKS, _EMBEDDINGS)
    assert _milvus(runtime_dir).count() == 3
    assert _milvus(runtime_dir, reset=True).count() == 0


def test_milvus_rejects_mismatched_lengths(runtime_dir: Path) -> None:
    store = _milvus(runtime_dir)
    with pytest.raises(ValueError):
        store.add(_CHUNKS, _EMBEDDINGS[:2])


def test_milvus_empty_add_and_non_positive_top_k(runtime_dir: Path) -> None:
    store = _milvus(runtime_dir)
    store.add([], [])
    assert store.count() == 0

    store.add(_CHUNKS, _EMBEDDINGS)
    assert store.query([1.0, 0.0, 0.0, 0.0], top_k=0) == []


def test_milvus_rejects_empty_uri() -> None:
    with pytest.raises(ValueError):
        MilvusVectorStore("   ", collection_name="x")


def test_factory_dispatches_by_backend(runtime_dir: Path) -> None:
    chroma_settings = Settings(
        _env_file=None,
        vector_backend="chroma",
        chroma_dir=runtime_dir / "chroma",
        chroma_collection="factory_test",
    )
    milvus_settings = Settings(
        _env_file=None,
        vector_backend="milvus",
        milvus_uri=str(runtime_dir / "factory_lite.db"),
        milvus_collection="factory_test",
    )

    assert isinstance(create_vector_store(chroma_settings), ChromaVectorStore)
    assert isinstance(create_vector_store(milvus_settings), MilvusVectorStore)


def test_factory_overrides_are_applied(runtime_dir: Path) -> None:
    """评测轨要建独立索引，覆盖项必须真的生效，否则会污染开发用集合。"""
    settings = Settings(
        _env_file=None,
        vector_backend="milvus",
        milvus_uri=str(runtime_dir / "override_lite.db"),
        milvus_collection="default_collection",
    )
    store = create_vector_store(settings, collection_name="eval_collection", reset=True)
    store.add(_CHUNKS, _EMBEDDINGS)

    assert store.count() == 3
    # 默认集合不应被写入
    default_store = create_vector_store(settings)
    assert default_store.count() == 0


def test_milvus_endpoint_env_var_avoids_pymilvus_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """环境变量必须叫 MILVUS_ENDPOINT，不能叫 MILVUS_URI。

    pymilvus 在 ``pymilvus/settings.py`` 里 import 时就 ``os.getenv("MILVUS_URI")``，
    并在 ``orm/connections.py`` 强制按 ``http[s]://host:port`` 解析。我们要支持
    Milvus Lite 的本地文件路径，用同名变量会在 import 阶段直接抛
    ``ConnectionConfigException``，而且报错点离配置很远、很难定位。
    """
    monkeypatch.setenv("MILVUS_ENDPOINT", "data/from_env.db")
    monkeypatch.setenv("MILVUS_URI", "data/should_be_ignored.db")

    settings = Settings()
    assert settings.milvus_uri == "data/from_env.db"


def test_settings_rejects_empty_milvus_uri_when_selected() -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, vector_backend="milvus", milvus_uri="  ")


def test_settings_rejects_empty_milvus_collection() -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, milvus_collection="  ")
