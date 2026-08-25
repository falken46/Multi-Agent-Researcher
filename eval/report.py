"""Render evidence-only Markdown reports from computed evaluation summaries."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.metrics import aggregate_retrieval_records, aggregate_task_records
from eval.models import (
    RetrievalGroupSummary,
    RetrievalObservation,
    TaskGroupSummary,
    TaskObservation,
    TaskTrack,
)


def render_markdown_report(
    *,
    retrieval_summaries: Sequence[RetrievalGroupSummary] = (),
    task_summaries: Sequence[TaskGroupSummary] = (),
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Render only supplied, computed values; never create placeholder metrics."""

    retrieval = tuple(retrieval_summaries)
    tasks = tuple(task_summaries)
    if not retrieval and not tasks:
        raise ValueError("at least one computed summary is required")
    for summary in retrieval:
        _validate_retrieval_summary(summary)
    for summary in tasks:
        _validate_task_summary(summary)

    lines = [
        "# DeepResearch Agent 评测报告",
        "",
        "> 本报告仅呈现结构化评测原始记录与 trace 汇总计算得到的指标；",
        "> 缺少结构化证据的实验会整节省略，不生成空白值或占位值。",
    ]
    if metadata:
        lines.extend(_render_metadata(metadata))
    if retrieval:
        lines.extend(_render_retrieval_table(retrieval))
        lines.extend(_render_retrieval_attribution(retrieval))

    by_track: defaultdict[TaskTrack, list[TaskGroupSummary]] = defaultdict(list)
    for summary in tasks:
        by_track[summary.track].append(summary)
    for track in ("P", "Q"):
        if track in by_track:
            lines.extend(_render_task_table(track, by_track[track]))
            lines.extend(_render_task_attribution(track, by_track[track]))

    lines.extend(
        [
            "",
            "## 指标边界",
            "",
            "- Coverage 是 Unicode/空白归一化后的关键词或同义短语覆盖率，不代表语义正确性。",
            "- Citation validity 只验证报告中的数字引用编号是否存在于输入引用集合，不证明来源支持对应结论。",
            "- token、成本、调用次数、fallback、revision 与耗时均来自结构化 trace summary，不从普通日志提取。",
            "- Critic 自身评分不作为 Critic 有效性的独立质量证据。",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown_report(
    path: Path | str,
    *,
    retrieval_summaries: Sequence[RetrievalGroupSummary] = (),
    task_summaries: Sequence[TaskGroupSummary] = (),
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write an evidence-only report and return its resolved input path."""

    output_path = Path(path)
    content = render_markdown_report(
        retrieval_summaries=retrieval_summaries,
        task_summaries=task_summaries,
        metadata=metadata,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8", newline="\n")
    return output_path


def load_jsonl_records(paths: Sequence[Path | str]) -> tuple[dict[str, Any], ...]:
    """读取结构化 raw 证据；空行和非对象记录均视为输入错误。"""

    if not paths:
        raise ValueError("at least one raw JSONL path is required")
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"raw JSONL file not found: {path}")
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                raise ValueError(f"blank JSONL line: {path}:{line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL record: {path}:{line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record must be an object: {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError("raw JSONL files must contain records")
    return tuple(records)


def write_retrieval_report_from_raw(
    output_path: Path | str,
    *,
    raw_paths: Sequence[Path | str],
    overwrite: bool = False,
) -> Path:
    """从 R 轨 raw 记录计算指标并生成可追踪的 Markdown 报告。"""

    return write_report_from_raw(
        output_path,
        retrieval_raw_paths=raw_paths,
        overwrite=overwrite,
    )


def write_task_report_from_raw(
    output_path: Path | str,
    *,
    raw_paths: Sequence[Path | str],
    overwrite: bool = False,
) -> Path:
    """从 P/Q raw 与其中的 trace summary 生成 Markdown 报告。"""

    return write_report_from_raw(
        output_path,
        task_raw_paths=raw_paths,
        overwrite=overwrite,
    )


def write_report_from_raw(
    output_path: Path | str,
    *,
    retrieval_raw_paths: Sequence[Path | str] = (),
    task_raw_paths: Sequence[Path | str] = (),
    overwrite: bool = False,
) -> Path:
    """合并已有 R/P/Q raw；未提供证据的轨道不会出现在报告中。"""

    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"evaluation report already exists: {output}")
    if not retrieval_raw_paths and not task_raw_paths:
        raise ValueError("at least one retrieval or task raw path is required")

    retrieval_summaries: tuple[RetrievalGroupSummary, ...] = ()
    task_summaries: tuple[TaskGroupSummary, ...] = ()
    metadata: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds")
    }
    if retrieval_raw_paths:
        retrieval_records = load_jsonl_records(retrieval_raw_paths)
        retrieval_observations = tuple(
            RetrievalObservation.from_raw(item) for item in retrieval_records
        )
        retrieval_summaries = aggregate_retrieval_records(retrieval_observations)
        metadata.update(
            _retrieval_metadata(
                retrieval_records,
                raw_paths=retrieval_raw_paths,
            )
        )
    if task_raw_paths:
        task_records = load_jsonl_records(task_raw_paths)
        task_observations = tuple(
            TaskObservation.from_raw(item) for item in task_records
        )
        task_summaries = aggregate_task_records(task_observations)
        metadata.update(_task_metadata(task_records, raw_paths=task_raw_paths))
    return write_markdown_report(
        output,
        retrieval_summaries=retrieval_summaries,
        task_summaries=task_summaries,
        metadata=metadata,
    )


def _render_metadata(metadata: Mapping[str, Any]) -> list[str]:
    lines = ["", "## 实验元信息", "", "| 字段 | 值 |", "|---|---|"]
    for key in sorted(metadata, key=str):
        normalized_key = str(key).strip()
        if not normalized_key:
            raise ValueError("metadata keys must not be empty")
        value = metadata[key]
        if value is None:
            raise ValueError(f"metadata value must not be null: {normalized_key}")
        rendered = _redacted_metadata_value(normalized_key, value)
        lines.append(f"| {_cell(normalized_key)} | {_cell(rendered)} |")
    return lines


def _render_retrieval_table(
    summaries: Sequence[RetrievalGroupSummary],
) -> list[str]:
    lines = [
        "",
        "## R 轨：检索质量",
        "",
        "| 组 | 题目数 | 观测数 | Candidate Recall@20 | Hit@5 | MRR@5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in sorted(summaries, key=lambda item: item.group):
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(summary.group),
                    str(summary.unique_case_count),
                    str(summary.observation_count),
                    _percentage(summary.candidate_recall_at_20),
                    _percentage(summary.hit_at_5),
                    f"{summary.mrr_at_5:.4f}",
                ]
            )
            + " |"
        )
    return lines


def _render_retrieval_attribution(
    summaries: Sequence[RetrievalGroupSummary],
) -> list[str]:
    by_group = {summary.group: summary for summary in summaries}
    if not {"R1", "R2", "R3", "R4"}.issubset(by_group):
        return []
    r1 = by_group["R1"]
    r2 = by_group["R2"]
    r3 = by_group["R3"]
    r4 = by_group["R4"]
    return [
        "",
        "## R 轨归因分析",
        "",
        "- R3 相较 R1："
        f"Candidate Recall@20 {_signed_pp(r3.candidate_recall_at_20 - r1.candidate_recall_at_20)}，"
        f"Hit@5 {_signed_pp(r3.hit_at_5 - r1.hit_at_5)}，"
        f"MRR@5 {_signed_decimal(r3.mrr_at_5 - r1.mrr_at_5)}。",
        "- R3 相较 R2："
        f"Candidate Recall@20 {_signed_pp(r3.candidate_recall_at_20 - r2.candidate_recall_at_20)}，"
        f"Hit@5 {_signed_pp(r3.hit_at_5 - r2.hit_at_5)}，"
        f"MRR@5 {_signed_decimal(r3.mrr_at_5 - r2.mrr_at_5)}。",
        "- R4 相较 R3："
        f"Candidate Recall@20 {_signed_pp(r4.candidate_recall_at_20 - r3.candidate_recall_at_20)}，"
        f"Hit@5 {_signed_pp(r4.hit_at_5 - r3.hit_at_5)}，"
        f"MRR@5 {_signed_decimal(r4.mrr_at_5 - r3.mrr_at_5)}。",
        "- 上述差值是当前固定子集上的描述性结果，不代表统计显著性；负向结果同样保留。",
    ]


def _render_task_table(
    track: TaskTrack,
    summaries: Sequence[TaskGroupSummary],
) -> list[str]:
    title = "并发微基准" if track == "P" else "反思质量与代价"
    lines = [
        "",
        f"## {track} 轨：{title}",
        "",
        "| 组 | 题目数 | 观测数 | Completion | Coverage | Citation validity | 平均耗时 (ms) | P95 (ms) | 平均 token | 平均成本 | 平均 LLM 调用 | Fallback 触发率 | Revision 触发率 | 平均返工次数 | 价格版本 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for summary in sorted(summaries, key=lambda item: item.group):
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(summary.group),
                    str(summary.unique_case_count),
                    str(summary.observation_count),
                    _percentage(summary.completion_rate),
                    _percentage(summary.mean_coverage),
                    _percentage(summary.mean_citation_validity),
                    f"{summary.mean_latency_ms:.3f}",
                    f"{summary.p95_latency_ms:.3f}",
                    f"{summary.mean_tokens:.2f}",
                    f"{summary.mean_cost:.10f} {_cell(summary.currency)}",
                    f"{summary.mean_llm_calls:.2f}",
                    _percentage(summary.fallback_trigger_rate),
                    _percentage(summary.revision_trigger_rate),
                    f"{summary.mean_revision_count:.2f}",
                    _cell(summary.pricing_version),
                ]
            )
            + " |"
        )
    return lines


def _render_task_attribution(
    track: TaskTrack,
    summaries: Sequence[TaskGroupSummary],
) -> list[str]:
    by_group = {summary.group: summary for summary in summaries}
    if track == "P" and {"P1", "P2"}.issubset(by_group):
        baseline = by_group["P1"]
        experiment = by_group["P2"]
        return [
            "",
            "### P 轨配对差值",
            "",
            "- P2 相较 P1："
            f"平均耗时 {_signed_percentage_change(experiment.mean_latency_ms, baseline.mean_latency_ms)}，"
            f"P95 {_signed_percentage_change(experiment.p95_latency_ms, baseline.p95_latency_ms)}，"
            f"平均 token {_signed_percentage_change(experiment.mean_tokens, baseline.mean_tokens)}。",
            "- P1/P2 均关闭 Critic；差异只用于描述当前固定任务上的并发效果。",
        ]
    if track == "Q" and {"Q1", "Q2"}.issubset(by_group):
        baseline = by_group["Q1"]
        experiment = by_group["Q2"]
        return [
            "",
            "### Q 轨配对差值",
            "",
            "- Q2 相较 Q1："
            f"Completion {_signed_pp(experiment.completion_rate - baseline.completion_rate)}，"
            f"Coverage {_signed_pp(experiment.mean_coverage - baseline.mean_coverage)}，"
            f"Citation validity {_signed_pp(experiment.mean_citation_validity - baseline.mean_citation_validity)}。",
            "- 资源代价："
            f"平均耗时 {_signed_percentage_change(experiment.mean_latency_ms, baseline.mean_latency_ms)}，"
            f"平均 token {_signed_percentage_change(experiment.mean_tokens, baseline.mean_tokens)}，"
            f"平均成本 {_signed_percentage_change(experiment.mean_cost, baseline.mean_cost)}。",
            "- Critic 分数是过程自评，不作为独立质量结论；负向差值同样保留。",
        ]
    return []


def _validate_retrieval_summary(summary: RetrievalGroupSummary) -> None:
    if not isinstance(summary, RetrievalGroupSummary):
        raise TypeError("retrieval summaries must be RetrievalGroupSummary values")
    _validate_counts(summary.observation_count, summary.unique_case_count)
    if not summary.group.strip():
        raise ValueError("retrieval summary group must not be empty")
    _validate_unit_interval(
        "candidate_recall_at_20", summary.candidate_recall_at_20
    )
    _validate_unit_interval("hit_at_5", summary.hit_at_5)
    _validate_unit_interval("mrr_at_5", summary.mrr_at_5)


def _validate_task_summary(summary: TaskGroupSummary) -> None:
    if not isinstance(summary, TaskGroupSummary):
        raise TypeError("task summaries must be TaskGroupSummary values")
    _validate_counts(summary.observation_count, summary.unique_case_count)
    for name, value in (
        ("completion_rate", summary.completion_rate),
        ("mean_coverage", summary.mean_coverage),
        ("mean_citation_validity", summary.mean_citation_validity),
        ("fallback_trigger_rate", summary.fallback_trigger_rate),
        ("revision_trigger_rate", summary.revision_trigger_rate),
    ):
        _validate_unit_interval(name, value)
    for name, value in (
        ("mean_latency_ms", summary.mean_latency_ms),
        ("p95_latency_ms", summary.p95_latency_ms),
        ("mean_tokens", summary.mean_tokens),
        ("mean_cost", summary.mean_cost),
        ("mean_llm_calls", summary.mean_llm_calls),
        ("mean_revision_count", summary.mean_revision_count),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise ValueError(f"{name} must be non-negative")
    if summary.track not in {"P", "Q"}:
        raise ValueError("task summary track must be 'P' or 'Q'")
    if not summary.group.strip() or not summary.currency.strip():
        raise ValueError("task summary group and currency must not be empty")
    if not summary.pricing_version.strip():
        raise ValueError("pricing_version must not be empty")


def _validate_counts(observation_count: int, unique_case_count: int) -> None:
    if (
        isinstance(observation_count, bool)
        or isinstance(unique_case_count, bool)
        or not isinstance(observation_count, int)
        or not isinstance(unique_case_count, int)
        or observation_count <= 0
        or unique_case_count <= 0
    ):
        raise ValueError("summary counts must be positive")
    if unique_case_count > observation_count:
        raise ValueError("unique_case_count must not exceed observation_count")


def _validate_unit_interval(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be between zero and one")


def _percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _redacted_metadata_value(key: str, value: Any) -> str:
    lowered = key.casefold()
    if any(
        marker in lowered
        for marker in ("api_key", "secret", "password", "access_token", "auth_token")
    ):
        return "[REDACTED]"
    return str(value)


def _retrieval_metadata(
    records: Sequence[Mapping[str, Any]],
    *,
    raw_paths: Sequence[Path | str],
) -> dict[str, Any]:
    groups = sorted({_required_record_text(record, "group") for record in records})
    case_ids = {_required_record_text(record, "case_id") for record in records}
    metadata = {
        "candidate_k": _single_value(records, "candidate_k"),
        "dataset": _single_value(records, "dataset"),
        "embedding_backend": _single_value(records, "embedding_backend"),
        "embedding_model": _single_value(records, "embedding_model"),
        "final_k": _single_value(records, "final_k"),
        "groups": ", ".join(groups),
        "observation_count": len(records),
        "query_count": len(case_ids),
        "raw_files": ", ".join(Path(path).as_posix() for path in raw_paths),
    }
    fusion_records = [
        record for record in records if record.get("group") in {"R3", "R4"}
    ]
    if fusion_records:
        metadata["rrf_k"] = _single_value(fusion_records, "rrf_k")
    rerank_records = [record for record in records if record.get("group") == "R4"]
    if rerank_records:
        metadata["rerank_backend"] = _single_value(
            rerank_records,
            "rerank_backend",
        )
        metadata["rerank_model"] = _single_value(rerank_records, "rerank_model")
    return metadata


def _task_metadata(
    records: Sequence[Mapping[str, Any]],
    *,
    raw_paths: Sequence[Path | str],
) -> dict[str, Any]:
    groups = sorted({_required_record_text(record, "group") for record in records})
    case_ids = {_required_record_text(record, "case_id") for record in records}
    tracks = sorted({_required_record_text(record, "track") for record in records})
    return {
        "task_cache_mode": _single_value(records, "cache_mode"),
        "task_cache_snapshot_id": _single_value(records, "cache_snapshot_id"),
        "task_case_count": len(case_ids),
        "task_groups": ", ".join(groups),
        "task_model_name": _single_value(records, "model_name"),
        "task_observation_count": len(records),
        "task_planner_mode": _single_value(records, "planner_mode"),
        "task_raw_files": ", ".join(Path(path).as_posix() for path in raw_paths),
        "task_search_provider": _single_value(records, "search_provider"),
        "task_tracks": ", ".join(tracks),
    }


def _single_value(records: Sequence[Mapping[str, Any]], name: str) -> Any:
    values: list[Any] = []
    for record in records:
        if name not in record or record[name] is None:
            raise ValueError(f"raw record is missing {name}")
        value = record[name]
        if value not in values:
            values.append(value)
    if len(values) != 1:
        raise ValueError(f"raw records mix {name} values")
    return values[0]


def _required_record_text(record: Mapping[str, Any], name: str) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"raw retrieval record requires non-empty {name}")
    return value.strip()


def _signed_pp(value: float) -> str:
    return f"{value * 100:+.2f} 个百分点"


def _signed_decimal(value: float) -> str:
    return f"{value:+.4f}"


def _signed_percentage_change(value: float, baseline: float) -> str:
    if baseline == 0:
        return "N/A（基线为 0）"
    return f"{(value / baseline - 1) * 100:+.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an evidence-only Phase 13 Markdown report from raw JSONL."
    )
    parser.add_argument("--retrieval-raw", nargs="+", type=Path)
    parser.add_argument("--task-raw", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("eval/reports/comparison.md"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = write_report_from_raw(
        args.out,
        retrieval_raw_paths=args.retrieval_raw or (),
        task_raw_paths=args.task_raw or (),
        overwrite=args.overwrite,
    )
    print(output.as_posix())


__all__ = [
    "load_jsonl_records",
    "render_markdown_report",
    "write_markdown_report",
    "write_report_from_raw",
    "write_retrieval_report_from_raw",
    "write_task_report_from_raw",
]


if __name__ == "__main__":
    main()
