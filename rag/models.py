"""检索层跨模块共享的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field

MetadataValue = str | int | float | bool
Metadata = dict[str, MetadataValue]


@dataclass(frozen=True)
class Document:
    """加载器产出的完整文档或 PDF 单页。"""

    text: str
    metadata: Metadata


@dataclass(frozen=True)
class Chunk:
    """可被两路索引共享的最小文本单元。"""

    id: str
    text: str
    metadata: Metadata


@dataclass(frozen=True)
class RetrievalResult:
    """检索通道、融合层与工具层之间的统一结果。"""

    chunk_id: str
    text: str
    source: str
    chunk_index: int
    score: float
    channel: str
    metadata: Metadata = field(default_factory=dict)


__all__ = [
    "Chunk",
    "Document",
    "Metadata",
    "MetadataValue",
    "RetrievalResult",
]
