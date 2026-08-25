"""Markdown、纯文本与 PDF 知识库文档加载器。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from rag.models import Document

logger = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


@dataclass(frozen=True)
class LoadFailure:
    path: str
    error: str


@dataclass(frozen=True)
class LoadReport:
    documents: list[Document]
    failures: list[LoadFailure]


class DocumentLoadError(RuntimeError):
    """单个知识库文档无法解析。"""


def load_file(path: Path, *, source_path: str | None = None) -> list[Document]:
    """加载一个受支持的文件；PDF 按页产出文档。"""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    normalized_source = source_path or file_path.as_posix()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentLoadError(f"unsupported file type: {suffix or '<none>'}")

    try:
        if suffix == ".pdf":
            return _load_pdf(file_path, normalized_source)
        text = file_path.read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeError) as exc:
        raise DocumentLoadError(f"failed to read {file_path}: {exc}") from exc

    if not text:
        raise DocumentLoadError(f"document is empty: {file_path}")
    return [
        Document(
            text=text,
            metadata={"source_path": normalized_source, "file_type": suffix[1:]},
        )
    ]


def load_directory(directory: Path) -> LoadReport:
    """递归加载目录，单文件失败只进入失败清单，不中断其他文件。"""
    root = Path(directory)
    if not root.is_dir():
        raise DocumentLoadError(f"knowledge base directory not found: {root}")

    documents: list[Document] = []
    failures: list[LoadFailure] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        source_path = path.relative_to(root).as_posix()
        logger.info("kb loader input path=%s", source_path)
        try:
            loaded = load_file(path, source_path=source_path)
            documents.extend(loaded)
            logger.info("kb loader output path=%s documents=%s", source_path, len(loaded))
        except DocumentLoadError as exc:
            failures.append(LoadFailure(path=source_path, error=str(exc)))
            logger.warning("kb loader skipped path=%s error=%s", source_path, exc)
    return LoadReport(documents=documents, failures=failures)


def _load_pdf(path: Path, source_path: str) -> list[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentLoadError("PDF support requires the pypdf package") from exc

    try:
        reader = PdfReader(path)
        documents = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "source_path": source_path,
                        "file_type": "pdf",
                        "page": page_number,
                    },
                )
            )
    except Exception as exc:
        raise DocumentLoadError(f"failed to parse {path}: {exc}") from exc

    if not documents:
        raise DocumentLoadError(f"PDF contains no extractable text: {path}")
    return documents


__all__ = [
    "DocumentLoadError",
    "LoadFailure",
    "LoadReport",
    "load_directory",
    "load_file",
]
