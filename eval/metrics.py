"""Deterministic metrics over structured evaluation and trace records."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import fmean
from typing import Any

from eval.models import (
    CoverConcept,
    RetrievalCaseMetrics,
    RetrievalGroupSummary,
    RetrievalObservation,
    TaskCaseMetrics,
    TaskGroupSummary,
    TaskObservation,
    TaskStatus,
    TaskTrack,
)

_CITATION_PATTERN = re.compile(r"\[(\d+)\](?!\s*:)")


def candidate_recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    gold_chunk_ids: Sequence[str],
    *,
    k: int,
) -> float:
    """Return the fraction of distinct gold chunks present in top ``k``."""

    ranked = _ranked_unique(retrieved_chunk_ids)
    gold = _gold_set(gold_chunk_ids)
    _require_positive_k(k)
    return len(gold.intersection(ranked[:k])) / len(gold)


def hit_at_k(
    retrieved_chunk_ids: Sequence[str],
    gold_chunk_ids: Sequence[str],
    *,
    k: int,
) -> float:
    """Return one when any gold chunk occurs in top ``k``, otherwise zero."""

    ranked = _ranked_unique(retrieved_chunk_ids)
    gold = _gold_set(gold_chunk_ids)
    _require_positive_k(k)
    return float(any(chunk_id in gold for chunk_id in ranked[:k]))


def mrr_at_k(
    retrieved_chunk_ids: Sequence[str],
    gold_chunk_ids: Sequence[str],
    *,
    k: int,
) -> float:
    """Return reciprocal rank of the first gold chunk, truncated at ``k``."""

    ranked = _ranked_unique(retrieved_chunk_ids)
    gold = _gold_set(gold_chunk_ids)
    _require_positive_k(k)
    for rank, chunk_id in enumerate(ranked[:k], start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def candidate_recall_at_20(
    retrieved_chunk_ids: Sequence[str],
    gold_chunk_ids: Sequence[str],
) -> float:
    return candidate_recall_at_k(retrieved_chunk_ids, gold_chunk_ids, k=20)


def hit_at_5(
    retrieved_chunk_ids: Sequence[str],
    gold_chunk_ids: Sequence[str],
) -> float:
    return hit_at_k(retrieved_chunk_ids, gold_chunk_ids, k=5)


def mrr_at_5(
    retrieved_chunk_ids: Sequence[str],
    gold_chunk_ids: Sequence[str],
) -> float:
    return mrr_at_k(retrieved_chunk_ids, gold_chunk_ids, k=5)


def evaluate_retrieval_case(
    case: RetrievalObservation | Mapping[str, Any],
) -> RetrievalCaseMetrics:
    """Evaluate one canonical R-track raw observation."""

    observation = _retrieval_observation(case)
    return RetrievalCaseMetrics(
        case_id=observation.case_id,
        group=observation.group,
        round_index=observation.round_index,
        candidate_recall_at_20=candidate_recall_at_20(
            observation.candidate_chunk_ids,
            observation.gold_chunk_ids,
        ),
        hit_at_5=hit_at_5(
            observation.retrieved_chunk_ids,
            observation.gold_chunk_ids,
        ),
        mrr_at_5=mrr_at_5(
            observation.retrieved_chunk_ids,
            observation.gold_chunk_ids,
        ),
    )


def aggregate_retrieval_group(
    group: str,
    cases: Iterable[RetrievalObservation | Mapping[str, Any]],
) -> RetrievalGroupSummary:
    """Macro-average R-track observations belonging to one group."""

    normalized = [_retrieval_observation(case) for case in cases]
    if not normalized:
        raise ValueError("retrieval group must contain at least one observation")
    if any(case.group != group for case in normalized):
        raise ValueError("retrieval observations contain a different group")
    _require_unique_runs(normalized)
    metrics = [evaluate_retrieval_case(case) for case in normalized]
    return RetrievalGroupSummary(
        group=group,
        observation_count=len(metrics),
        unique_case_count=len({item.case_id for item in metrics}),
        candidate_recall_at_20=fmean(
            item.candidate_recall_at_20 for item in metrics
        ),
        hit_at_5=fmean(item.hit_at_5 for item in metrics),
        mrr_at_5=fmean(item.mrr_at_5 for item in metrics),
    )


def aggregate_retrieval_records(
    cases: Iterable[RetrievalObservation | Mapping[str, Any]],
) -> tuple[RetrievalGroupSummary, ...]:
    """Group structured R-track raw observations without reading log text."""

    by_group: defaultdict[str, list[RetrievalObservation]] = defaultdict(list)
    for case in cases:
        observation = _retrieval_observation(case)
        by_group[observation.group].append(observation)
    if not by_group:
        raise ValueError("retrieval records must not be empty")
    return tuple(
        aggregate_retrieval_group(group, by_group[group])
        for group in sorted(by_group)
    )


def completion_score(final_report: str, *, status: TaskStatus) -> float:
    """A task completes only with explicit success and a non-empty report."""

    if not isinstance(final_report, str):
        raise ValueError("final_report must be a string")
    if status not in {"completed", "failed"}:
        raise ValueError("status must be 'completed' or 'failed'")
    return float(status == "completed" and bool(final_report.strip()))


def coverage_score(
    final_report: str,
    must_cover: Sequence[CoverConcept | Sequence[str] | str],
) -> float:
    """Measure concept coverage using any-of alternatives per concept.

    This is intentionally a weak lexical metric.  Unicode and whitespace are
    normalized, but no semantic correctness is inferred.
    """

    if not isinstance(final_report, str):
        raise ValueError("final_report must be a string")
    concepts = _normalize_cover_concepts(must_cover)
    normalized_report = _normalize_match_text(final_report)
    matched = sum(
        any(_normalize_match_text(term) in normalized_report for term in concept)
        for concept in concepts
    )
    return matched / len(concepts)


def extract_citation_ids(final_report: str) -> tuple[str, ...]:
    """Extract numeric inline citation occurrences such as ``[1]``.

    Markdown reference-definition labels (``[1]: ...``) are excluded.  The
    returned tuple keeps duplicate occurrences because validity is measured per
    citation use, not merely per distinct identifier.
    """

    if not isinstance(final_report, str):
        raise ValueError("final_report must be a string")
    return tuple(_CITATION_PATTERN.findall(final_report))


def citation_validity_score(
    final_report: str,
    valid_citation_ids: Sequence[str | int],
) -> float:
    """Return the fraction of cited numeric IDs present in the supplied source set.

    A report with no inline citation receives zero rather than an artificial
    perfect score.  This metric checks identifier validity only; it does not
    claim that a cited source semantically entails the surrounding statement.
    """

    valid = _normalize_citation_ids(valid_citation_ids)
    cited = extract_citation_ids(final_report)
    if not cited:
        return 0.0
    return sum(citation_id in valid for citation_id in cited) / len(cited)


def evaluate_task_case(
    case: TaskObservation | Mapping[str, Any],
) -> TaskCaseMetrics:
    """Evaluate one P/Q task using raw output plus a trace summary."""

    observation = _task_observation(case)
    return TaskCaseMetrics(
        case_id=observation.case_id,
        group=observation.group,
        track=observation.track,
        round_index=observation.round_index,
        completion=completion_score(
            observation.final_report,
            status=observation.status,
        ),
        coverage=coverage_score(observation.final_report, observation.must_cover),
        citation_validity=citation_validity_score(
            observation.final_report,
            observation.valid_citation_ids,
        ),
        total_latency_ms=observation.trace.total_latency_ms,
        total_tokens=observation.trace.total_tokens,
        total_cost=observation.trace.total_cost,
        llm_calls=observation.trace.llm_calls,
        fallback_count=observation.trace.fallback_count,
        revision_count=observation.trace.revision_count,
    )


def aggregate_task_group(
    track: TaskTrack,
    group: str,
    cases: Iterable[TaskObservation | Mapping[str, Any]],
) -> TaskGroupSummary:
    """Macro-average one P/Q group and aggregate only trace-derived resources."""

    normalized = [_task_observation(case) for case in cases]
    if not normalized:
        raise ValueError("task group must contain at least one observation")
    if track not in {"P", "Q"}:
        raise ValueError("track must be 'P' or 'Q'")
    if any(case.track != track or case.group != group for case in normalized):
        raise ValueError("task observations contain a different track or group")
    _require_unique_runs(normalized)

    currencies = {case.trace.currency for case in normalized}
    pricing_versions = {case.trace.pricing_version for case in normalized}
    if len(currencies) != 1 or len(pricing_versions) != 1:
        raise ValueError("task group mixes currency or pricing versions")

    metrics = [evaluate_task_case(case) for case in normalized]
    latencies = [item.total_latency_ms for item in metrics]
    return TaskGroupSummary(
        track=track,
        group=group,
        observation_count=len(metrics),
        unique_case_count=len({item.case_id for item in metrics}),
        completion_rate=fmean(item.completion for item in metrics),
        mean_coverage=fmean(item.coverage for item in metrics),
        mean_citation_validity=fmean(item.citation_validity for item in metrics),
        mean_latency_ms=fmean(latencies),
        p95_latency_ms=_nearest_rank_percentile(latencies, 0.95),
        mean_tokens=fmean(item.total_tokens for item in metrics),
        mean_cost=fmean(item.total_cost for item in metrics),
        mean_llm_calls=fmean(item.llm_calls for item in metrics),
        fallback_trigger_rate=fmean(
            float(item.fallback_count > 0) for item in metrics
        ),
        revision_trigger_rate=fmean(
            float(item.revision_count > 0) for item in metrics
        ),
        mean_revision_count=fmean(item.revision_count for item in metrics),
        currency=next(iter(currencies)),
        pricing_version=next(iter(pricing_versions)),
    )


def aggregate_task_records(
    cases: Iterable[TaskObservation | Mapping[str, Any]],
) -> tuple[TaskGroupSummary, ...]:
    """Group P/Q raw observations by their explicit track and group fields."""

    grouped: defaultdict[tuple[TaskTrack, str], list[TaskObservation]] = defaultdict(
        list
    )
    for case in cases:
        observation = _task_observation(case)
        grouped[(observation.track, observation.group)].append(observation)
    if not grouped:
        raise ValueError("task records must not be empty")
    return tuple(
        aggregate_task_group(track, group, grouped[(track, group)])
        for track, group in sorted(grouped)
    )


def _retrieval_observation(
    case: RetrievalObservation | Mapping[str, Any],
) -> RetrievalObservation:
    if isinstance(case, RetrievalObservation):
        return case
    if isinstance(case, Mapping):
        return RetrievalObservation.from_raw(case)
    raise TypeError("retrieval case must be an observation or mapping")


def _task_observation(
    case: TaskObservation | Mapping[str, Any],
) -> TaskObservation:
    if isinstance(case, TaskObservation):
        return case
    if isinstance(case, Mapping):
        return TaskObservation.from_raw(case)
    raise TypeError("task case must be an observation or mapping")


def _ranked_unique(chunk_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(chunk_ids, str):
        raise ValueError("retrieved_chunk_ids must be a sequence of IDs")
    ranked: list[str] = []
    seen: set[str] = set()
    for chunk_id in chunk_ids:
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("retrieved_chunk_ids must contain non-empty strings")
        normalized = chunk_id.strip()
        if normalized not in seen:
            seen.add(normalized)
            ranked.append(normalized)
    return tuple(ranked)


def _gold_set(chunk_ids: Sequence[str]) -> frozenset[str]:
    if isinstance(chunk_ids, str):
        raise ValueError("gold_chunk_ids must be a sequence of IDs")
    gold: set[str] = set()
    for chunk_id in chunk_ids:
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("gold_chunk_ids must contain non-empty strings")
        gold.add(chunk_id.strip())
    if not gold:
        raise ValueError("gold_chunk_ids must not be empty")
    return frozenset(gold)


def _normalize_cover_concepts(
    concepts: Sequence[CoverConcept | Sequence[str] | str],
) -> tuple[CoverConcept, ...]:
    if isinstance(concepts, str) or not concepts:
        raise ValueError("must_cover must contain at least one concept")
    normalized: list[CoverConcept] = []
    for concept in concepts:
        alternatives: Sequence[str] = (concept,) if isinstance(concept, str) else concept
        if isinstance(alternatives, str) or not alternatives:
            raise ValueError("each must_cover concept must contain alternatives")
        terms = tuple(
            term.strip()
            for term in alternatives
            if isinstance(term, str) and term.strip()
        )
        if len(terms) != len(alternatives):
            raise ValueError("must_cover alternatives must be non-empty strings")
        normalized.append(terms)
    return tuple(normalized)


def _normalize_match_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(normalized.split())


def _normalize_citation_ids(values: Sequence[str | int]) -> frozenset[str]:
    if isinstance(values, str):
        raise ValueError("valid_citation_ids must be a sequence")
    normalized: set[str] = set()
    for value in values:
        if isinstance(value, bool):
            raise ValueError("valid_citation_ids must contain positive integers")
        citation_id = str(value).strip()
        if not citation_id.isdigit() or int(citation_id) <= 0:
            raise ValueError("valid_citation_ids must contain positive integers")
        normalized.add(citation_id)
    return frozenset(normalized)


def _require_positive_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")


def _nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile values must not be empty")
    if not 0 < percentile <= 1:
        raise ValueError("percentile must satisfy 0 < percentile <= 1")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _require_unique_runs(
    observations: Sequence[RetrievalObservation | TaskObservation],
) -> None:
    keys = [(item.case_id, item.round_index) for item in observations]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate case_id and round_index in one group")


__all__ = [
    "aggregate_retrieval_group",
    "aggregate_retrieval_records",
    "aggregate_task_group",
    "aggregate_task_records",
    "candidate_recall_at_20",
    "candidate_recall_at_k",
    "citation_validity_score",
    "completion_score",
    "coverage_score",
    "evaluate_retrieval_case",
    "evaluate_task_case",
    "extract_citation_ids",
    "hit_at_5",
    "hit_at_k",
    "mrr_at_5",
    "mrr_at_k",
]
