"""预热 Phase 13 冻结 Planner 子问题的 Web query cache。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.orchestration_runner import (
    DEFAULT_DATASET_PATH,
    DEFAULT_WEB_CACHE_DIR,
    prewarm_initial_query_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prewarm initial Web queries without running any LLM."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_WEB_CACHE_DIR)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-queries", type=int, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    report = prewarm_initial_query_cache(
        snapshot_id=args.snapshot_id,
        max_queries=args.max_queries,
        dataset_path=args.dataset,
        cache_dir=args.cache_dir,
        case_ids=args.case_ids,
        case_limit=args.limit,
        live=args.live,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
