"""运行 Phase 13 的 P/Q 编排评测。"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from eval.orchestration_runner import (
    DEFAULT_DATASET_PATH,
    DEFAULT_GROUPS,
    DEFAULT_WEB_CACHE_DIR,
    default_raw_output_path,
    run_orchestration_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fixed-Planner P/Q orchestration evaluation tasks."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_WEB_CACHE_DIR)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--cache-mode", choices=("record", "replay-only"), required=True)
    parser.add_argument("--groups", nargs="+", choices=DEFAULT_GROUPS, required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    report = asyncio.run(
        run_orchestration_evaluation(
            output_path=args.output or default_raw_output_path(),
            snapshot_id=args.snapshot_id,
            cache_mode=args.cache_mode,
            max_tasks=args.max_tasks,
            dataset_path=args.dataset,
            cache_dir=args.cache_dir,
            groups=args.groups,
            rounds=args.rounds,
            case_ids=args.case_ids,
            case_limit=args.limit,
            resume=args.resume,
            live=args.live,
        )
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
