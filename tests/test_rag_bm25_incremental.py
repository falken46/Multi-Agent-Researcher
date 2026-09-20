"""BM25 增量索引测试（Phase 20）。

重点验证三件事：

1. `extend` 的结果与「全量重建」等价 —— 增量不能换来错误的检索结果
2. 分词确实被复用了（这才是增量省下来的东西）
3. v1 旧索引能自动升级，不是硬报错
"""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import patch

import pytest

from rag.bm25 import INDEX_VERSION, BM25Index
from rag.models import Chunk

_DOCS = [
    Chunk(id="c1", text="RRF 融合只依赖名次，不依赖分数", metadata={"source_path": "a.md", "chunk_index": 0}),
    Chunk(id="c2", text="BM25 擅长专有名词与编号的精确匹配", metadata={"source_path": "b.md", "chunk_index": 1}),
    Chunk(id="c3", text="向量检索擅长语义相近但用词不同的情况", metadata={"source_path": "c.md", "chunk_index": 2}),
]


def _index(runtime_dir: Path, name: str = "index.pkl") -> BM25Index:
    return BM25Index(runtime_dir / "bm25" / name)


def test_extend_matches_full_rebuild(runtime_dir: Path) -> None:
    """增量追加与全量重建必须给出相同的检索结果。

    这是本次改造的核心正确性要求：省掉重复分词，但不能改变排序或分数。
    """
    incremental = _index(runtime_dir, "incremental.pkl")
    incremental.add(_DOCS[:1])
    incremental.extend(_DOCS[1:])

    rebuilt = _index(runtime_dir, "rebuilt.pkl")
    rebuilt.add(_DOCS)

    assert incremental.count() == rebuilt.count() == 3
    for query in ("RRF 名次", "BM25 编号", "语义相近"):
        left = incremental.query(query, top_k=5)
        right = rebuilt.query(query, top_k=5)
        assert [hit.chunk_id for hit in left] == [hit.chunk_id for hit in right]
        for a, b in zip(left, right):
            assert a.score == pytest.approx(b.score)


def test_extend_only_tokenizes_new_documents(runtime_dir: Path) -> None:
    """已有文档不重新分词 —— 这正是增量省下来的开销。"""
    index = _index(runtime_dir)
    index.add(_DOCS)

    with patch("rag.bm25.tokenize", wraps=__import__("rag.bm25", fromlist=["tokenize"]).tokenize) as spy:
        tokenized = index.extend([
            Chunk(id="c4", text="重排是精排，召回是粗筛", metadata={"source_path": "d.md", "chunk_index": 3})
        ])

    assert tokenized == 1
    assert spy.call_count == 1  # 只切了新文档，三篇旧文档没再切
    assert index.count() == 4


def test_extend_upserts_by_chunk_id(runtime_dir: Path) -> None:
    index = _index(runtime_dir)
    index.add(_DOCS)

    changed = Chunk(id="c2", text="完全换掉的新内容 关于 Milvus", metadata={"source_path": "b.md", "chunk_index": 1})
    index.extend([changed])

    assert index.count() == 3  # 覆盖而不是新增
    hits = index.query("Milvus", top_k=5)
    assert [hit.chunk_id for hit in hits] == ["c2"]
    # 旧内容不该再被检索到
    assert index.query("专有名词 编号", top_k=5) == [] or all(
        hit.chunk_id != "c2" for hit in index.query("专有名词 编号", top_k=5)
    )


def test_extend_skips_tokenizing_unchanged_text(runtime_dir: Path) -> None:
    """正文没变、只改 metadata 时，连分词都不用做。"""
    index = _index(runtime_dir)
    index.add(_DOCS)

    same_text_new_meta = Chunk(
        id="c1", text=_DOCS[0].text, metadata={"source_path": "a.md", "chunk_index": 99}
    )
    tokenized = index.extend([same_text_new_meta])

    assert tokenized == 0
    assert index.count() == 3


def test_extend_empty_is_noop(runtime_dir: Path) -> None:
    index = _index(runtime_dir)
    index.add(_DOCS)
    assert index.extend([]) == 0
    assert index.count() == 3


def test_reload_does_not_retokenize(runtime_dir: Path) -> None:
    """从磁盘加载 v2 索引时不再跑 jieba —— v1 每次加载都要把全语料切一遍。"""
    path = runtime_dir / "bm25" / "index.pkl"
    BM25Index(path).add(_DOCS)

    with patch("rag.bm25.tokenize") as spy:
        reloaded = BM25Index(path)
    assert spy.call_count == 0
    assert reloaded.count() == 3
    # 加载后仍可正常检索（引擎是从缓存 token 重建的）
    assert [hit.chunk_id for hit in reloaded.query("RRF 名次", top_k=3)] == ["c1"]


def test_legacy_v1_index_is_upgraded_not_rejected(runtime_dir: Path) -> None:
    """v1 索引缺 token 缓存，但数据完整 —— 应补算升级而不是报错。"""
    path = runtime_dir / "bm25" / "legacy.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy_payload = {
        "version": 1,
        "chunks": [
            {"id": c.id, "text": c.text, "metadata": dict(c.metadata)} for c in _DOCS
        ],
    }
    with path.open("wb") as handle:
        pickle.dump(legacy_payload, handle)

    index = BM25Index(path)
    assert index.count() == 3
    assert [hit.chunk_id for hit in index.query("BM25 编号", top_k=3)] == ["c2"]

    # 触发一次写入后，磁盘上应升级为 v2 并带上 token 缓存
    index.extend([Chunk(id="c9", text="新增文档", metadata={})])
    with path.open("rb") as handle:
        upgraded = pickle.load(handle)
    assert upgraded["version"] == INDEX_VERSION
    assert len(upgraded["tokens"]) == 4


def test_unknown_version_still_fails_loudly(runtime_dir: Path) -> None:
    """未知格式必须报错 —— 自动升级只对已知的 v1 生效，不能吞掉真正的格式问题。"""
    path = runtime_dir / "bm25" / "future.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"version": 999, "chunks": []}, handle)

    with pytest.raises(RuntimeError):
        BM25Index(path)
