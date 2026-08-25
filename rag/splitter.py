"""适合中文技术文档的确定性递归字符切分。"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from rag.models import Chunk, Document

SEPARATORS: tuple[str, ...] = ("\n\n", "\n", "。", "！", "？", "；", "，", " ")


def split_documents(
    documents: Sequence[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """按优先级寻找自然边界，并为相邻切片保留字符重叠。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    chunks: list[Chunk] = []
    for document in documents:
        text = document.text.strip()
        if not text:
            continue
        source = str(document.metadata.get("source_path", "unknown"))
        page = document.metadata.get("page", "")
        doc_id = _stable_id(f"{source}\n{page}\n{text}")
        for chunk_index, chunk_text in enumerate(
            _split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "source_path": source,
                }
            )
            chunk_id = _stable_id(f"{doc_id}:{chunk_index}:{chunk_text}")
            chunks.append(Chunk(id=chunk_id, text=chunk_text, metadata=metadata))
    return chunks


def _split_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        hard_end = min(start + chunk_size, text_length)
        if hard_end == text_length:
            end = hard_end
        else:
            end = _preferred_boundary(text, start, hard_end)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= text_length:
            break
        next_start = max(0, end - chunk_overlap)
        start = next_start if next_start > start else end
    return pieces


def _preferred_boundary(text: str, start: int, hard_end: int) -> int:
    """递归字符切分的核心：依优先级寻找窗口后半段的分隔符。"""
    minimum = start + max(1, (hard_end - start) // 2)
    for separator in SEPARATORS:
        index = text.rfind(separator, minimum, hard_end)
        if index >= minimum:
            return index + len(separator)
    return hard_end


def _stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


__all__ = ["SEPARATORS", "split_documents"]
