from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.models import RetrievalGroupSummary, TaskGroupSummary
from eval.report import (
    render_markdown_report,
    write_markdown_report,
    write_report_from_raw,
    write_retrieval_report_from_raw,
)


def test_report_renders_only_computed_groups_and_actual_values() -> None:
    content = render_markdown_report(
        retrieval_summaries=[_retrieval_summary()],
        task_summaries=[_task_summary("P", "P2")],
        metadata={
            "dataset_version": "phase13-v1",
            "git_commit": "deadbeef",
            "api_key": "must-not-leak",
        },
    )

    assert "R 轨：检索质量" in content
    assert "P 轨：并发微基准" in content
    assert "Q 轨：" not in content
    assert "75.00%" in content
    assert "0.6250" in content
    assert "1500.000" in content
    assert "0.0125000000 CNY" in content
    assert "test-v1" in content
    assert "must-not-leak" not in content
    assert "[REDACTED]" in content
    assert "待填" not in content
    assert "__" not in content


def test_report_preserves_metric_boundaries() -> None:
    content = render_markdown_report(task_summaries=[_task_summary("Q", "Q1")])

    assert "Citation validity" in content
    assert "不证明来源支持对应结论" in content
    assert "Critic 自身评分不作为" in content
    assert "普通日志" in content


def test_report_rejects_empty_or_invalid_summaries() -> None:
    with pytest.raises(ValueError, match="computed summary"):
        render_markdown_report()

    invalid = RetrievalGroupSummary(
        group="R1",
        observation_count=1,
        unique_case_count=1,
        candidate_recall_at_20=1.1,
        hit_at_5=1.0,
        mrr_at_5=1.0,
        recall_at_5=1.0,
        ndcg_at_5=1.0,
        map_at_20=1.0,
    )
    with pytest.raises(ValueError, match="candidate_recall"):
        render_markdown_report(retrieval_summaries=[invalid])


def test_write_markdown_report_creates_utf8_artifact(runtime_dir: Path) -> None:
    path = runtime_dir / "reports" / "comparison.md"

    returned = write_markdown_report(
        path,
        retrieval_summaries=[_retrieval_summary()],
    )

    assert returned == path
    assert path.read_text(encoding="utf-8").startswith(
        "# DeepResearch Agent 评测报告"
    )


def test_write_retrieval_report_from_raw_computes_deltas(
    runtime_dir: Path,
) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    raw_path = runtime_dir / "raw.jsonl"
    records = []
    retrieved_by_group = {
        "R1": ["gold"],
        "R2": ["noise", "gold"],
        "R3": ["gold", "noise"],
        "R4": ["noise", "gold"],
    }
    for group, retrieved in retrieved_by_group.items():
        ranked = retrieved + [
            chunk_id for chunk_id in ["gold", "noise"] if chunk_id not in retrieved
        ]
        records.append(
            {
                "case_id": "case-1",
                "group": group,
                "candidate_chunk_ids": ["gold", "noise"],
                "ranked_chunk_ids": ranked,
                "retrieved_chunk_ids": retrieved,
                "gold_chunk_ids": ["gold"],
                "dataset": "test-dataset",
                "candidate_k": 20,
                "final_k": 5,
                "embedding_backend": "fake",
                "embedding_model": "fake-model",
                "rrf_k": 60 if group in {"R3", "R4"} else None,
                "rerank_backend": "onnx" if group == "R4" else "none",
                "rerank_model": "fake-reranker" if group == "R4" else "",
            }
        )
    raw_path.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )
    report_path = runtime_dir / "report.md"

    write_retrieval_report_from_raw(report_path, raw_paths=[raw_path])

    content = report_path.read_text(encoding="utf-8")
    assert "R 轨归因分析" in content
    assert "R4 相较 R3" in content
    assert "-0.5000" in content
    assert "负向结果同样保留" in content
    assert "fake-reranker" in content


def test_retrieval_report_rejects_raw_without_full_ranked_list(
    runtime_dir: Path,
) -> None:
    """Legacy raw only stored the final slice, so MAP@20 would silently be AP@5."""

    runtime_dir.mkdir(parents=True, exist_ok=True)
    raw_path = runtime_dir / "legacy.jsonl"
    raw_path.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "group": "R4",
                "candidate_chunk_ids": ["gold", "noise"],
                "retrieved_chunk_ids": ["gold"],
                "gold_chunk_ids": ["gold"],
                "dataset": "test-dataset",
                "candidate_k": 20,
                "final_k": 5,
                "embedding_backend": "fake",
                "embedding_model": "fake-model",
                "rrf_k": 60,
                "rerank_backend": "onnx",
                "rerank_model": "fake-reranker",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ranked_chunk_ids"):
        write_retrieval_report_from_raw(
            runtime_dir / "report.md",
            raw_paths=[raw_path],
        )


def test_combined_report_reads_task_metrics_only_from_raw_trace_summary(
    runtime_dir: Path,
) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    task_path = runtime_dir / "task.jsonl"
    task_records = [
        _task_raw(group="P1", track="P", latency_ms=200.0, tokens=100),
        _task_raw(group="P2", track="P", latency_ms=100.0, tokens=100),
        _task_raw(group="Q1", track="Q", latency_ms=100.0, tokens=100),
        _task_raw(
            group="Q2",
            track="Q",
            latency_ms=150.0,
            tokens=120,
            report="关键点 [1]",
            revision_count=1,
        ),
    ]
    task_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in task_records),
        encoding="utf-8",
    )
    output = runtime_dir / "combined.md"

    write_report_from_raw(output, task_raw_paths=[task_path])

    content = output.read_text(encoding="utf-8")
    assert "P 轨：并发微基准" in content
    assert "Q 轨：反思质量与代价" in content
    assert "P2 相较 P1" in content
    assert "平均耗时 -50.00%" in content
    assert "Q2 相较 Q1" in content
    assert "平均 token +20.00%" in content
    assert "snapshot-test" in content


def _retrieval_summary() -> RetrievalGroupSummary:
    return RetrievalGroupSummary(
        group="R2-hybrid",
        observation_count=30,
        unique_case_count=30,
        candidate_recall_at_20=0.75,
        hit_at_5=0.8,
        mrr_at_5=0.625,
        recall_at_5=0.7,
        ndcg_at_5=0.66,
        map_at_20=0.58,
    )


def _task_summary(track: str, group: str) -> TaskGroupSummary:
    return TaskGroupSummary(
        track=track,
        group=group,
        observation_count=30,
        unique_case_count=15,
        completion_rate=0.9,
        mean_coverage=0.8,
        mean_citation_validity=0.95,
        mean_latency_ms=1500.0,
        p95_latency_ms=2200.0,
        mean_tokens=1234.5,
        mean_cost=0.0125,
        mean_llm_calls=4.0,
        fallback_trigger_rate=0.2,
        revision_trigger_rate=0.4,
        mean_revision_count=0.5,
        currency="CNY",
        pricing_version="test-v1",
    )


def _task_raw(
    *,
    group: str,
    track: str,
    latency_ms: float,
    tokens: int,
    report: str = "关键点 [1]",
    revision_count: int = 0,
) -> dict[str, object]:
    return {
        "case_id": "case-1",
        "group": group,
        "track": track,
        "round_index": 1,
        "final_report": report,
        "must_cover": [["关键点"]],
        "valid_citation_ids": ["1"],
        "status": "completed",
        "cache_mode": "replay-only",
        "cache_snapshot_id": "snapshot-test",
        "model_name": "fake-model",
        "planner_mode": "fixed_dataset",
        "search_provider": "tavily",
        "trace_summary": {
            "total_latency_ms": latency_ms,
            "total_tokens": tokens,
            "total_cost": tokens / 100000,
            "llm_calls": 2,
            "fallback_count": 1,
            "revision_count": revision_count,
            "currency": "CNY",
            "pricing_version": "test-v1",
        },
    }
