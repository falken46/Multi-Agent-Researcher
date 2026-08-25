"""本地知识库建库命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.config import get_settings
from rag.pipeline import build_index


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 DeepResearch 本地知识库索引")
    parser.add_argument("--dir", type=Path, help="知识库语料目录，默认读取 KB_DIR")
    parser.add_argument(
        "--embedding-backend",
        choices=("fastembed", "remote", "fake"),
        help="仅覆盖本次建库使用的 embedding 后端",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.embedding_backend:
        settings = settings.model_copy(
            update={"embedding_backend": args.embedding_backend}
        )
    report = build_index(args.dir, settings=settings)
    print(
        "建库完成："
        f"文档 {report.document_count}，切片 {report.chunk_count}，"
        f"向量 {report.vector_count}，BM25 {report.bm25_count}"
    )
    for failure in report.load_failures:
        print(f"跳过文件：{failure.path} | {failure.error}")
    for channel, error in report.channel_errors.items():
        print(f"通道降级：{channel} | {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
