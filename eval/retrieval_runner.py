"""Phase 13 R1-R4 离线检索评测 runner。

评测直接复用产品检索层的 embedding、Chroma、BM25、RRF 与 rerank 原语，
但使用独立索引目录，并且不会触发 LLM、Web 搜索或 fallback。
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from core.config import Settings, get_settings
from eval.metrics import aggregate_retrieval_records
from eval.models import RetrievalGroupSummary, RetrievalObservation
from eval.retrieval_dataset import (
    DATASET_NAME,
    DEFAULT_OUTPUT_DIR as DEFAULT_DATASET_DIR,
    PreparedRetrievalDataset,
    RetrievalCase,
    load_prepared_dataset,
)
from rag.bm25 import BM25Index
from rag.embeddings import EmbeddingBackend, create_embedding_backend
from rag.hybrid import rrf_fuse
from rag.models import RetrievalResult
from rag.rerank import rerank
from rag.vectorstore import ChromaVectorStore

RetrievalGroup = Literal["R1", "R2", "R3", "R4"]

DEFAULT_GROUPS: tuple[RetrievalGroup, ...] = ("R1", "R2", "R3", "R4")
DEFAULT_INDEX_DIR = Path("eval/.cache/retrieval_index")
DEFAULT_RAW_DIR = Path("eval/reports/raw")
CANDIDATE_K = 20
FINAL_K = 5

_VECTOR_GROUPS = frozenset({"R1", "R3", "R4"})
_BM25_GROUPS = frozenset({"R2", "R3", "R4"})


@dataclass(frozen=True)
class RetrievalRunReport:
    """一次 R 轨运行的可审计结果。"""

    output_path: Path
    query_count: int
    observation_count: int
    passage_count: int
    groups: tuple[RetrievalGroup, ...]
    summaries: tuple[RetrievalGroupSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path.as_posix(),
            "query_count": self.query_count,
            "observation_count": self.observation_count,
            "passage_count": self.passage_count,
            "groups": list(self.groups),
            "summaries": [asdict(summary) for summary in self.summaries],
        }


def default_raw_output_path() -> Path:
    """生成不会覆盖历史证据的 UTC 时间戳文件名。"""

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_RAW_DIR / f"retrieval_{timestamp}.jsonl"


def run_retrieval_evaluation(
    *,
    dataset_dir: Path | str = DEFAULT_DATASET_DIR,
    output_path: Path | str | None = None,
    index_dir: Path | str = DEFAULT_INDEX_DIR,
    groups: Sequence[str] = DEFAULT_GROUPS,
    settings: Settings | None = None,
    case_limit: int | None = None,
    overwrite: bool = False,
    embedding_backend: EmbeddingBackend | None = None,
) -> RetrievalRunReport:
    """在共享 passage 池上运行指定 R 组，并原子写入结构化 JSONL。"""

    normalized_groups = _normalize_groups(groups)
    current = settings or get_settings()
    _validate_backends(normalized_groups, current, embedding_backend)
    dataset = load_prepared_dataset(dataset_dir)
    cases = _select_cases(dataset, case_limit)
    output = Path(output_path) if output_path is not None else default_raw_output_path()
    if output.exists() and not overwrite:
        raise FileExistsError(f"retrieval raw output already exists: {output}")

    vector_store: ChromaVectorStore | None = None
    embedder: EmbeddingBackend | None = None
    bm25_index: BM25Index | None = None
    evaluation_index_dir = Path(index_dir)

    if _requires_vector(normalized_groups):
        embedder = embedding_backend or create_embedding_backend(current)
        vector_store = _build_vector_index(
            dataset,
            index_dir=evaluation_index_dir / "chroma",
            embedder=embedder,
        )
    if _requires_bm25(normalized_groups):
        bm25_index = _build_bm25_index(
            dataset,
            index_path=evaluation_index_dir / "bm25" / "index.pkl",
        )

    records: list[dict[str, Any]] = []
    for case in cases:
        records.extend(
            _evaluate_case(
                case,
                groups=normalized_groups,
                settings=current,
                embedder=embedder,
                vector_store=vector_store,
                bm25_index=bm25_index,
            )
        )

    for record in records:
        RetrievalObservation.from_raw(record)
    _write_jsonl_atomic(output, records, overwrite=overwrite)
    summaries = aggregate_retrieval_records(records)
    return RetrievalRunReport(
        output_path=output,
        query_count=len(cases),
        observation_count=len(records),
        passage_count=len(dataset.chunks),
        groups=normalized_groups,
        summaries=summaries,
    )


def _build_vector_index(
    dataset: PreparedRetrievalDataset,
    *,
    index_dir: Path,
    embedder: EmbeddingBackend,
) -> ChromaVectorStore:
    embeddings = embedder.embed_documents([chunk.text for chunk in dataset.chunks])
    store = ChromaVectorStore(
        index_dir,
        collection_name="phase13_t2_reranking",
        reset=True,
    )
    store.add(dataset.chunks, embeddings)
    return store


def _build_bm25_index(
    dataset: PreparedRetrievalDataset,
    *,
    index_path: Path,
) -> BM25Index:
    index = BM25Index(index_path)
    index.add(dataset.chunks)
    return index


def _evaluate_case(
    case: RetrievalCase,
    *,
    groups: tuple[RetrievalGroup, ...],
    settings: Settings,
    embedder: EmbeddingBackend | None,
    vector_store: ChromaVectorStore | None,
    bm25_index: BM25Index | None,
) -> list[dict[str, Any]]:
    channel_results: dict[str, list[RetrievalResult]] = {}
    channel_latency_ms: dict[str, float] = {}

    if _requires_vector(groups):
        assert embedder is not None and vector_store is not None
        started = perf_counter()
        query_embedding = embedder.embed_query(case.query)
        channel_results["vector"] = vector_store.query(
            query_embedding,
            top_k=CANDIDATE_K,
        )
        channel_latency_ms["vector"] = (perf_counter() - started) * 1000

    if _requires_bm25(groups):
        assert bm25_index is not None
        started = perf_counter()
        channel_results["bm25"] = bm25_index.query(
            case.query,
            top_k=CANDIDATE_K,
        )
        channel_latency_ms["bm25"] = (perf_counter() - started) * 1000

    fused: list[RetrievalResult] | None = None
    fusion_latency_ms = 0.0
    if any(group in {"R3", "R4"} for group in groups):
        started = perf_counter()
        fused = rrf_fuse(channel_results, rrf_k=settings.rrf_k)[:CANDIDATE_K]
        fusion_latency_ms = (perf_counter() - started) * 1000

    records: list[dict[str, Any]] = []
    for group in groups:
        if group == "R1":
            candidates = channel_results["vector"]
            ranked = candidates
            latency_ms = channel_latency_ms["vector"]
        elif group == "R2":
            candidates = channel_results["bm25"]
            ranked = candidates
            latency_ms = channel_latency_ms["bm25"]
        elif group == "R3":
            assert fused is not None
            candidates = fused
            ranked = candidates
            latency_ms = (
                channel_latency_ms["vector"]
                + channel_latency_ms["bm25"]
                + fusion_latency_ms
            )
        else:
            assert fused is not None
            candidates = fused
            started = perf_counter()
            # Rerank at candidate depth so rank-aware metrics (MAP@20) have a
            # full ordering.  ``rerank`` sorts then slices, so the top ``FINAL_K``
            # is identical to reranking with ``top_n=FINAL_K`` directly.
            ranked = rerank(
                case.query,
                candidates,
                top_n=CANDIDATE_K,
                settings=settings,
            )
            latency_ms = (
                channel_latency_ms["vector"]
                + channel_latency_ms["bm25"]
                + fusion_latency_ms
                + (perf_counter() - started) * 1000
            )
        records.append(
            _raw_record(
                case,
                group=group,
                candidates=candidates,
                ranked=ranked,
                retrieved=ranked[:FINAL_K],
                latency_ms=latency_ms,
                settings=settings,
            )
        )
    return records


def _raw_record(
    case: RetrievalCase,
    *,
    group: RetrievalGroup,
    candidates: Sequence[RetrievalResult],
    ranked: Sequence[RetrievalResult],
    retrieved: Sequence[RetrievalResult],
    latency_ms: float,
    settings: Settings,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "track": "R",
        "dataset": DATASET_NAME,
        "case_id": case.case_id,
        "query": case.query,
        "group": group,
        "round_index": 1,
        "candidate_k": CANDIDATE_K,
        "final_k": FINAL_K,
        "candidate_chunk_ids": [item.chunk_id for item in candidates],
        "ranked_chunk_ids": [item.chunk_id for item in ranked],
        "retrieved_chunk_ids": [item.chunk_id for item in retrieved],
        "gold_chunk_ids": list(case.gold_passage_ids),
        "source_candidate_chunk_ids": list(case.source_candidate_passage_ids),
        "candidate_results": [_result_evidence(item) for item in candidates],
        "retrieved_results": [_result_evidence(item) for item in retrieved],
        "latency_ms": latency_ms,
        "embedding_backend": settings.embedding_backend,
        "embedding_model": settings.embedding_model,
        "rerank_backend": settings.rerank_backend if group == "R4" else "none",
        "rerank_model": settings.rerank_model if group == "R4" else "",
        "rrf_k": settings.rrf_k if group in {"R3", "R4"} else None,
    }


def _result_evidence(result: RetrievalResult) -> dict[str, Any]:
    return {
        "chunk_id": result.chunk_id,
        "score": result.score,
        "score_kind": result.score_kind,
        "channel": result.channel,
        "fallback_confidence": result.fallback_confidence,
    }


def _select_cases(
    dataset: PreparedRetrievalDataset,
    case_limit: int | None,
) -> tuple[RetrievalCase, ...]:
    if case_limit is None:
        return dataset.cases
    if isinstance(case_limit, bool) or not isinstance(case_limit, int) or case_limit <= 0:
        raise ValueError("case_limit must be a positive integer")
    return dataset.cases[:case_limit]


def _normalize_groups(groups: Sequence[str]) -> tuple[RetrievalGroup, ...]:
    if isinstance(groups, str) or not groups:
        raise ValueError("groups must contain at least one R-track group")
    requested = {str(group).upper() for group in groups}
    unknown = requested.difference(DEFAULT_GROUPS)
    if unknown:
        raise ValueError(f"unknown retrieval groups: {', '.join(sorted(unknown))}")
    return tuple(group for group in DEFAULT_GROUPS if group in requested)


def _validate_backends(
    groups: tuple[RetrievalGroup, ...],
    settings: Settings,
    embedding_backend: EmbeddingBackend | None,
) -> None:
    if _requires_vector(groups) and embedding_backend is None:
        if settings.embedding_backend not in {"fastembed", "fake"}:
            raise ValueError("R-track only permits local fastembed or fake embeddings")
    if "R4" in groups and settings.rerank_backend != "onnx":
        raise ValueError("R4 requires rerank_backend='onnx'")


def _requires_vector(groups: Sequence[str]) -> bool:
    return any(group in _VECTOR_GROUPS for group in groups)


def _requires_bm25(groups: Sequence[str]) -> bool:
    return any(group in _BM25_GROUPS for group in groups)


def _write_jsonl_atomic(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    overwrite: bool,
) -> None:
    if not records:
        raise ValueError("retrieval raw records must not be empty")
    if path.exists() and not overwrite:
        raise FileExistsError(f"retrieval raw output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("x", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


__all__ = [
    "CANDIDATE_K",
    "DEFAULT_GROUPS",
    "DEFAULT_INDEX_DIR",
    "DEFAULT_RAW_DIR",
    "FINAL_K",
    "RetrievalGroup",
    "RetrievalRunReport",
    "default_raw_output_path",
    "run_retrieval_evaluation",
]
