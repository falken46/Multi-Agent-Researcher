"""下载并转换 Phase 13 使用的 T2Reranking 中文检索子集。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.retrieval_dataset import (
    DEFAULT_CACHE_PATH,
    DEFAULT_MAX_NEGATIVES,
    DEFAULT_MAX_POSITIVES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QUERY_COUNT,
    DEFAULT_SEED,
    download_source,
    iter_parquet_rows,
    prepare_retrieval_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a shared-corpus T2Reranking subset for Phase 13."
    )
    parser.add_argument("--queries", type=int, default=DEFAULT_QUERY_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-positives", type=int, default=DEFAULT_MAX_POSITIVES)
    parser.add_argument("--max-negatives", type=int, default=DEFAULT_MAX_NEGATIVES)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--redownload", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_path = download_source(args.cache, overwrite=args.redownload)
    report = prepare_retrieval_dataset(
        iter_parquet_rows(source_path),
        args.output,
        query_count=args.queries,
        seed=args.seed,
        max_positives=args.max_positives,
        max_negatives=args.max_negatives,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output_dir": report.output_dir.as_posix(),
                "query_count": report.query_count,
                "passage_count": report.passage_count,
                "positive_relation_count": report.positive_relation_count,
                "negative_relation_count": report.negative_relation_count,
                "seed": report.seed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
