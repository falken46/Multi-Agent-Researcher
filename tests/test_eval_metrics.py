from __future__ import annotations

import math

import pytest

from eval.metrics import (
    aggregate_retrieval_group,
    aggregate_retrieval_records,
    aggregate_task_group,
    average_precision_at_20,
    average_precision_at_k,
    candidate_recall_at_20,
    candidate_recall_at_k,
    citation_validity_score,
    completion_score,
    coverage_score,
    evaluate_retrieval_case,
    evaluate_task_case,
    extract_citation_ids,
    hit_at_5,
    mrr_at_5,
    ndcg_at_5,
    recall_at_5,
)
from eval.models import RetrievalObservation


def test_retrieval_metrics_use_candidate_and_top_five_cutoffs() -> None:
    retrieved = tuple(f"chunk-{index}" for index in range(1, 21))
    gold = ("chunk-3", "chunk-10", "missing")

    assert candidate_recall_at_20(retrieved, gold) == pytest.approx(2 / 3)
    assert hit_at_5(retrieved, gold) == 1.0
    assert mrr_at_5(retrieved, gold) == pytest.approx(1 / 3)


def test_retrieval_metrics_deduplicate_ranked_ids_and_reject_missing_gold() -> None:
    assert mrr_at_5(("noise", "noise", "gold"), ("gold",)) == 0.5

    with pytest.raises(ValueError, match="gold_chunk_ids"):
        candidate_recall_at_20(("chunk",), ())
    with pytest.raises(ValueError, match="positive"):
        candidate_recall_at_k(("chunk",), ("chunk",), k=0)


def test_retrieval_case_and_group_accept_structured_raw_records() -> None:
    records = [
        {
            "id": "RET-001",
            "group": "R1",
            "candidate_chunk_ids": ["gold-a", "gold-b", "noise"],
            "retrieved_chunk_ids": ["gold-a", "noise"],
            "gold_chunk_ids": ["gold-a", "gold-b"],
        },
        {
            "id": "RET-002",
            "group": "R1",
            "candidate_chunk_ids": ["noise", "gold-c"],
            "retrieved_chunk_ids": ["noise", "gold-c"],
            "gold_chunk_ids": ["gold-c"],
        },
    ]

    first = evaluate_retrieval_case(records[0])
    summary = aggregate_retrieval_group("R1", records)
    all_summaries = aggregate_retrieval_records(records)

    assert first.candidate_recall_at_20 == 1.0
    assert summary.observation_count == 2
    assert summary.unique_case_count == 2
    assert summary.candidate_recall_at_20 == 1.0
    assert summary.hit_at_5 == 1.0
    assert summary.mrr_at_5 == pytest.approx(0.75)
    assert all_summaries == (summary,)


def test_recall_and_ndcg_separate_multi_positive_ranking_from_first_hit() -> None:
    ranked = ("gold-a", "noise-1", "gold-b", "noise-2", "noise-3")
    gold = ("gold-a", "gold-b", "gold-c")

    # hit_at_5 saturates on the first relevant passage and cannot distinguish
    # a ranking that surfaces one gold chunk from one that surfaces two.
    assert hit_at_5(ranked, gold) == 1.0
    assert recall_at_5(ranked, gold) == pytest.approx(2 / 3)

    gain = 1 / math.log2(2) + 1 / math.log2(4)
    ideal = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_5(ranked, gold) == pytest.approx(gain / ideal)


def test_average_precision_normalises_by_truncated_gold_count() -> None:
    ranked = ("gold-a", "noise", "gold-b")
    gold = ("gold-a", "gold-b")

    assert average_precision_at_20(ranked, gold) == pytest.approx((1.0 + 2 / 3) / 2)

    # A query holding more gold chunks than the cutoff can still reach 1.0.
    assert average_precision_at_k(("g1", "g2"), ("g1", "g2", "g3"), k=2) == 1.0


def test_ranked_chunk_ids_drive_rank_aware_metrics_and_must_extend_retrieved() -> None:
    raw = {
        "id": "RET-001",
        "group": "R4",
        "candidate_chunk_ids": ["a", "b", "c", "d"],
        "ranked_chunk_ids": ["c", "a", "d", "b"],
        "retrieved_chunk_ids": ["c", "a"],
        "gold_chunk_ids": ["a", "d"],
    }

    metrics = evaluate_retrieval_case(raw)

    # MAP@20 sees the full reranked ordering, including the gold chunk at rank 3
    # that the persisted top-2 slice alone would have hidden.
    assert metrics.recall_at_5 == pytest.approx(1.0)
    assert metrics.map_at_20 == pytest.approx((1 / 2 + 2 / 3) / 2)

    with pytest.raises(ValueError, match="prefix"):
        RetrievalObservation.from_raw({**raw, "retrieved_chunk_ids": ["a", "c"]})


def test_rank_aware_metrics_fall_back_to_retrieved_ids_for_legacy_raw() -> None:
    legacy = {
        "id": "RET-002",
        "group": "R3",
        "candidate_chunk_ids": ["a", "b", "c"],
        "retrieved_chunk_ids": ["b", "a"],
        "gold_chunk_ids": ["a"],
    }

    observation = RetrievalObservation.from_raw(legacy)

    assert observation.ranked_chunk_ids == ()
    assert observation.ranking_chunk_ids == ("b", "a")
    assert evaluate_retrieval_case(legacy).map_at_20 == pytest.approx(0.5)


def test_retrieval_raw_requires_final_results_to_come_from_candidates() -> None:
    raw = {
        "id": "RET-001",
        "group": "R1",
        "candidate_chunk_ids": ["candidate"],
        "retrieved_chunk_ids": ["not-a-candidate"],
        "gold_chunk_ids": ["candidate"],
    }

    with pytest.raises(ValueError, match="subset"):
        evaluate_retrieval_case(raw)


def test_coverage_supports_any_of_synonyms_and_unicode_normalization() -> None:
    report = "RRF 只依赖排名，因此无需 归一化；ＡＰＩ 使用异步并发。"
    must_cover = [
        ["排名而非分数", "只依赖排名"],
        ["免归一化", "无需归一化"],
        ["串行执行", "顺序执行"],
    ]

    assert coverage_score(report, must_cover) == pytest.approx(2 / 3)

    with pytest.raises(ValueError, match="must_cover"):
        coverage_score(report, [])


def test_citation_validity_counts_inline_occurrences_not_definitions() -> None:
    report = "结论一 [1]，结论二 [9]，再次引用 [1]。\n\n[1]: source"

    assert extract_citation_ids(report) == ("1", "9", "1")
    assert citation_validity_score(report, (1, 2)) == pytest.approx(2 / 3)
    assert citation_validity_score("没有引用", (1,)) == 0.0


def test_completion_requires_explicit_success_and_non_empty_report() -> None:
    assert completion_score("报告", status="completed") == 1.0
    assert completion_score("报告", status="failed") == 0.0
    assert completion_score("  ", status="completed") == 0.0


def test_task_case_reads_quality_from_raw_and_resources_from_trace_summary() -> None:
    metrics = evaluate_task_case(
        _task_raw(
            case_id="QA-001",
            group="Q1",
            report="包含覆盖点 [1] 与伪造来源 [7]",
            status="completed",
            latency_ms=1250.0,
            total_tokens=240,
            total_cost=0.003,
            llm_calls=3,
            fallback_count=1,
            revision_count=1,
        )
    )

    assert metrics.completion == 1.0
    assert metrics.coverage == 1.0
    assert metrics.citation_validity == 0.5
    assert metrics.total_latency_ms == 1250.0
    assert metrics.total_tokens == 240
    assert metrics.revision_count == 1


def test_task_group_macro_averages_quality_and_trace_metrics() -> None:
    records = [
        _task_raw(
            case_id="QA-001",
            group="Q1",
            report="包含覆盖点 [1]",
            latency_ms=100.0,
            total_tokens=100,
            total_cost=0.01,
            llm_calls=2,
            fallback_count=0,
            revision_count=0,
        ),
        _task_raw(
            case_id="QA-002",
            group="Q1",
            report="缺少关键词 [8]",
            status="failed",
            latency_ms=300.0,
            total_tokens=300,
            total_cost=0.03,
            llm_calls=4,
            fallback_count=2,
            revision_count=2,
        ),
    ]

    summary = aggregate_task_group("Q", "Q1", records)

    assert summary.observation_count == 2
    assert summary.unique_case_count == 2
    assert summary.completion_rate == 0.5
    assert summary.mean_coverage == 0.5
    assert summary.mean_citation_validity == 0.5
    assert summary.mean_latency_ms == 200.0
    assert summary.p95_latency_ms == 300.0
    assert summary.mean_tokens == 200.0
    assert summary.mean_cost == pytest.approx(0.02)
    assert summary.mean_llm_calls == 3.0
    assert summary.fallback_trigger_rate == 0.5
    assert summary.revision_trigger_rate == 0.5
    assert summary.mean_revision_count == 1.0
    assert summary.currency == "CNY"


def test_task_records_fail_closed_on_incomplete_or_mixed_trace_evidence() -> None:
    incomplete = _task_raw(case_id="QA-001", group="Q1", report="覆盖点 [1]")
    del incomplete["trace_summary"]["total_tokens"]

    with pytest.raises(ValueError, match="total_tokens"):
        evaluate_task_case(incomplete)

    first = _task_raw(case_id="QA-001", group="Q1", report="覆盖点 [1]")
    second = _task_raw(case_id="QA-002", group="Q1", report="覆盖点 [1]")
    second["trace_summary"]["pricing_version"] = "different"

    with pytest.raises(ValueError, match="pricing"):
        aggregate_task_group("Q", "Q1", [first, second])

    with pytest.raises(ValueError, match="duplicate"):
        aggregate_task_group("Q", "Q1", [first, first])


def _task_raw(
    *,
    case_id: str,
    group: str,
    report: str,
    status: str = "completed",
    latency_ms: float = 100.0,
    total_tokens: int = 100,
    total_cost: float = 0.01,
    llm_calls: int = 2,
    fallback_count: int = 0,
    revision_count: int = 0,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "group": group,
        "track": "Q",
        "round_index": 1,
        "final_report": report,
        "must_cover": [["覆盖点", "同义覆盖"]],
        "valid_citation_ids": [1],
        "status": status,
        "trace_summary": {
            "total_latency_ms": latency_ms,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "llm_calls": llm_calls,
            "fallback_count": fallback_count,
            "revision_count": revision_count,
            "currency": "CNY",
            "pricing_version": "test-v1",
        },
    }
