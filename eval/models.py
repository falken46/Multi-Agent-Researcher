"""Offline evaluation records and metric result models.

The models in this module form the boundary between JSONL runner output and the
pure metric/reporting layer.  They deliberately contain no network or log-file
access so every reported value must originate from a structured raw record or
from ``core.trace.summarize`` output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, TypeAlias

TaskTrack: TypeAlias = Literal["P", "Q"]
TaskStatus: TypeAlias = Literal["completed", "failed"]
CoverConcept: TypeAlias = tuple[str, ...]


@dataclass(frozen=True)
class TraceMetrics:
    """The resource and behaviour fields consumed from a trace summary."""

    total_latency_ms: float
    total_tokens: int
    total_cost: float
    llm_calls: int
    fallback_count: int
    revision_count: int
    currency: str
    pricing_version: str

    def __post_init__(self) -> None:
        _require_non_negative_float("total_latency_ms", self.total_latency_ms)
        _require_non_negative_int("total_tokens", self.total_tokens)
        _require_non_negative_float("total_cost", self.total_cost)
        _require_non_negative_int("llm_calls", self.llm_calls)
        _require_non_negative_int("fallback_count", self.fallback_count)
        _require_non_negative_int("revision_count", self.revision_count)
        _require_text("currency", self.currency)
        _require_text("pricing_version", self.pricing_version)

    @classmethod
    def from_summary(cls, summary: Mapping[str, Any]) -> TraceMetrics:
        """Create metrics from ``core.trace.summarize`` compatible data.

        Required fields fail closed instead of silently becoming zero.  This
        prevents an incomplete trace from looking like a free or instant run.
        """

        return cls(
            total_latency_ms=_required_float(summary, "total_latency_ms"),
            total_tokens=_required_int(summary, "total_tokens"),
            total_cost=_required_float(summary, "total_cost"),
            llm_calls=_required_int(summary, "llm_calls"),
            fallback_count=_required_int(summary, "fallback_count"),
            revision_count=_required_int(summary, "revision_count"),
            currency=_required_text(summary, "currency"),
            pricing_version=_required_text(summary, "pricing_version"),
        )


@dataclass(frozen=True)
class RetrievalObservation:
    """One ranked retrieval result evaluated against frozen gold chunks."""

    case_id: str
    group: str
    candidate_chunk_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    gold_chunk_ids: tuple[str, ...]
    round_index: int = 1
    ranked_chunk_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text("case_id", self.case_id)
        _require_text("group", self.group)
        _require_positive_int("round_index", self.round_index)
        _require_id_sequence("candidate_chunk_ids", self.candidate_chunk_ids)
        _require_id_sequence("retrieved_chunk_ids", self.retrieved_chunk_ids)
        if not set(self.retrieved_chunk_ids).issubset(self.candidate_chunk_ids):
            raise ValueError(
                "retrieved_chunk_ids must be a subset of candidate_chunk_ids"
            )
        _require_id_sequence(
            "gold_chunk_ids",
            self.gold_chunk_ids,
            allow_empty=False,
        )
        _require_id_sequence("ranked_chunk_ids", self.ranked_chunk_ids)
        if self.ranked_chunk_ids:
            if not set(self.ranked_chunk_ids).issubset(self.candidate_chunk_ids):
                raise ValueError(
                    "ranked_chunk_ids must be a subset of candidate_chunk_ids"
                )
            prefix = self.ranked_chunk_ids[: len(self.retrieved_chunk_ids)]
            if prefix != self.retrieved_chunk_ids:
                raise ValueError(
                    "retrieved_chunk_ids must be a prefix of ranked_chunk_ids"
                )

    @property
    def ranking_chunk_ids(self) -> tuple[str, ...]:
        """Return the deepest ranked list available for rank-aware metrics.

        Older raw records only persisted the final top-``k`` slice, so they
        fall back to ``retrieved_chunk_ids``.  Metrics computed at a cutoff
        deeper than that slice are only meaningful once the runner emits
        ``ranked_chunk_ids``.
        """

        return self.ranked_chunk_ids or self.retrieved_chunk_ids

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> RetrievalObservation:
        """Parse the canonical retrieval JSONL record used by the runner."""

        return cls(
            case_id=_raw_case_id(raw),
            group=_required_text(raw, "group"),
            candidate_chunk_ids=_id_tuple(raw, "candidate_chunk_ids"),
            retrieved_chunk_ids=_id_tuple(raw, "retrieved_chunk_ids"),
            gold_chunk_ids=_id_tuple(raw, "gold_chunk_ids", allow_empty=False),
            round_index=_optional_positive_int(raw, "round_index", default=1),
            ranked_chunk_ids=(
                _id_tuple(raw, "ranked_chunk_ids")
                if "ranked_chunk_ids" in raw
                else ()
            ),
        )


@dataclass(frozen=True)
class TaskObservation:
    """One P/Q-track task result plus its structured trace summary."""

    case_id: str
    group: str
    track: TaskTrack
    final_report: str
    must_cover: tuple[CoverConcept, ...]
    valid_citation_ids: tuple[str, ...]
    status: TaskStatus
    trace: TraceMetrics
    round_index: int = 1

    def __post_init__(self) -> None:
        _require_text("case_id", self.case_id)
        _require_text("group", self.group)
        if self.track not in {"P", "Q"}:
            raise ValueError("track must be 'P' or 'Q'")
        if self.status not in {"completed", "failed"}:
            raise ValueError("status must be 'completed' or 'failed'")
        _require_positive_int("round_index", self.round_index)
        if not self.must_cover:
            raise ValueError("must_cover must contain at least one concept")
        for concept in self.must_cover:
            _require_id_sequence("must_cover concept", concept, allow_empty=False)
        _require_id_sequence("valid_citation_ids", self.valid_citation_ids)

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> TaskObservation:
        """Parse a raw task record without consulting console/application logs."""

        trace_summary = raw.get("trace_summary")
        if not isinstance(trace_summary, Mapping):
            raise ValueError("trace_summary must be a mapping")
        raw_track = _required_text(raw, "track").upper()
        raw_status = _required_text(raw, "status").lower()
        if raw_track not in {"P", "Q"}:
            raise ValueError("track must be 'P' or 'Q'")
        if raw_status not in {"completed", "failed"}:
            raise ValueError("status must be 'completed' or 'failed'")
        return cls(
            case_id=_raw_case_id(raw),
            group=_required_text(raw, "group"),
            track=raw_track,
            final_report=_required_string(raw, "final_report"),
            must_cover=_cover_concepts(raw.get("must_cover")),
            valid_citation_ids=_citation_id_tuple(raw.get("valid_citation_ids")),
            status=raw_status,
            trace=TraceMetrics.from_summary(trace_summary),
            round_index=_optional_positive_int(raw, "round_index", default=1),
        )


@dataclass(frozen=True)
class RetrievalCaseMetrics:
    """Per-observation R-track metrics."""

    case_id: str
    group: str
    round_index: int
    candidate_recall_at_20: float
    hit_at_5: float
    mrr_at_5: float
    recall_at_5: float
    ndcg_at_5: float
    map_at_20: float


@dataclass(frozen=True)
class RetrievalGroupSummary:
    """Macro-averaged R-track metrics for one configuration group."""

    group: str
    observation_count: int
    unique_case_count: int
    candidate_recall_at_20: float
    hit_at_5: float
    mrr_at_5: float
    recall_at_5: float
    ndcg_at_5: float
    map_at_20: float


@dataclass(frozen=True)
class TaskCaseMetrics:
    """Per-observation quality, efficiency and behaviour metrics."""

    case_id: str
    group: str
    track: TaskTrack
    round_index: int
    completion: float
    coverage: float
    citation_validity: float
    total_latency_ms: float
    total_tokens: int
    total_cost: float
    llm_calls: int
    fallback_count: int
    revision_count: int


@dataclass(frozen=True)
class TaskGroupSummary:
    """Macro quality and trace-derived resource metrics for a P/Q group."""

    track: TaskTrack
    group: str
    observation_count: int
    unique_case_count: int
    completion_rate: float
    mean_coverage: float
    mean_citation_validity: float
    mean_latency_ms: float
    p95_latency_ms: float
    mean_tokens: float
    mean_cost: float
    mean_llm_calls: float
    fallback_trigger_rate: float
    revision_trigger_rate: float
    mean_revision_count: float
    currency: str
    pricing_version: str


def _raw_case_id(raw: Mapping[str, Any]) -> str:
    value = raw.get("case_id", raw.get("id"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("case_id must be a non-empty string")
    return value.strip()


def _cover_concepts(raw: Any) -> tuple[CoverConcept, ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("must_cover must contain at least one concept")
    concepts: list[CoverConcept] = []
    for item in raw:
        alternatives = (item,) if isinstance(item, str) else item
        if not isinstance(alternatives, (list, tuple)) or not alternatives:
            raise ValueError("each must_cover concept must contain alternatives")
        normalized = tuple(
            alternative.strip()
            for alternative in alternatives
            if isinstance(alternative, str) and alternative.strip()
        )
        if len(normalized) != len(alternatives):
            raise ValueError("must_cover alternatives must be non-empty strings")
        concepts.append(normalized)
    return tuple(concepts)


def _citation_id_tuple(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple, set, frozenset)):
        raise ValueError("valid_citation_ids must be a sequence")
    values: list[str] = []
    for value in raw:
        if isinstance(value, bool):
            raise ValueError("valid_citation_ids must contain positive integers")
        normalized = str(value).strip()
        if not normalized.isdigit() or int(normalized) <= 0:
            raise ValueError("valid_citation_ids must contain positive integers")
        if normalized not in values:
            values.append(normalized)
    return tuple(values)


def _id_tuple(
    raw: Mapping[str, Any],
    name: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    value = raw.get(name)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    normalized = tuple(
        item.strip() for item in value if isinstance(item, str) and item.strip()
    )
    if len(normalized) != len(value) or (not allow_empty and not normalized):
        suffix = " and must not be empty" if not allow_empty else ""
        raise ValueError(f"{name} must contain non-empty strings{suffix}")
    return normalized


def _require_id_sequence(
    name: str,
    values: tuple[str, ...],
    *,
    allow_empty: bool = True,
) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be a tuple")
    if not allow_empty and not values:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must contain non-empty strings")


def _required_string(raw: Mapping[str, Any], name: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _required_text(raw_or_name: Mapping[str, Any] | str, name_or_value: str) -> str:
    if isinstance(raw_or_name, Mapping):
        name = name_or_value
        value = raw_or_name.get(name)
    else:
        name = raw_or_name
        value = name_or_value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _required_int(raw: Mapping[str, Any], name: str) -> int:
    value = raw.get(name)
    _require_non_negative_int(name, value)
    return value


def _required_float(raw: Mapping[str, Any], name: str) -> float:
    value = raw.get(name)
    _require_non_negative_float(name, value)
    return float(value)


def _optional_positive_int(
    raw: Mapping[str, Any],
    name: str,
    *,
    default: int,
) -> int:
    value = raw.get(name, default)
    _require_positive_int(name, value)
    return value


def _require_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_non_negative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_non_negative_float(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-negative finite number")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be a non-negative finite number")


def _require_text(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


__all__ = [
    "CoverConcept",
    "RetrievalCaseMetrics",
    "RetrievalGroupSummary",
    "RetrievalObservation",
    "TaskCaseMetrics",
    "TaskGroupSummary",
    "TaskObservation",
    "TaskStatus",
    "TaskTrack",
    "TraceMetrics",
]
