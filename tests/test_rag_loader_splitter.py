from __future__ import annotations

from pathlib import Path

from rag.loader import load_directory
from rag.models import Document
from rag.splitter import split_documents


def test_loader_keeps_good_files_when_one_file_fails(runtime_dir: Path) -> None:
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "good.md").write_text("# 可用文档\n\n这是正文。", encoding="utf-8")
    (runtime_dir / "broken.txt").write_bytes(b"\xff\xfe\x00")
    (runtime_dir / "ignored.csv").write_text("ignored", encoding="utf-8")

    report = load_directory(runtime_dir)

    assert len(report.documents) == 1
    assert report.documents[0].metadata["source_path"] == "good.md"
    assert [failure.path for failure in report.failures] == ["broken.txt"]


def test_splitter_is_deterministic_and_preserves_metadata() -> None:
    document = Document(
        text=("甲" * 60) + "。" + ("乙" * 60) + "。" + ("丙" * 40),
        metadata={"source_path": "demo.md"},
    )

    first = split_documents([document], chunk_size=80, chunk_overlap=12)
    second = split_documents([document], chunk_size=80, chunk_overlap=12)

    assert first == second
    assert len(first) >= 3
    assert first[0].text.endswith("。")
    assert all(len(chunk.text) <= 80 for chunk in first)
    assert [chunk.metadata["chunk_index"] for chunk in first] == list(
        range(len(first))
    )
    assert all(chunk.metadata["source_path"] == "demo.md" for chunk in first)
    assert len({chunk.metadata["doc_id"] for chunk in first}) == 1
    assert first[0].text[-12:] in first[1].text
