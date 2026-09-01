"""运行 Phase 13 的 R1-R4 离线检索评测。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.config import get_settings
from eval.retrieval_dataset import DEFAULT_OUTPUT_DIR
from eval.retrieval_runner import (
    DEFAULT_GROUPS,
    DEFAULT_INDEX_DIR,
    default_raw_output_path,
    run_retrieval_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local-only Phase 13 R1-R4 retrieval benchmark."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--groups", nargs="+", choices=DEFAULT_GROUPS, default=DEFAULT_GROUPS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--embedding-backend", choices=("fastembed", "fake"))
    parser.add_argument("--embedding-model")
    parser.add_argument("--rerank-model")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    updates = {
        key: value
        for key, value in {
            "embedding_backend": args.embedding_backend,
            "embedding_model": args.embedding_model,
            "rerank_model": args.rerank_model,
        }.items()
        if value is not None
    }
    settings = get_settings().model_copy(update=updates)
    output_path = args.output or default_raw_output_path()
    report = run_retrieval_evaluation(
        dataset_dir=args.dataset,
        output_path=output_path,
        index_dir=args.index_dir,
        groups=args.groups,
        settings=settings,
        case_limit=args.limit,
        overwrite=args.overwrite,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
