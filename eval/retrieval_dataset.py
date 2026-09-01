"""T2Reranking 公开数据集的下载、抽样与本地评测适配。

原始数据只在显式执行准备命令时联网下载。转换后的所有 query 共用同一个
passage 池，避免把每道题限制在各自候选列表内而退化为纯排序测试。
"""

from __future__ import annotations

import html
import json
import os
import random
import re
import shutil
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from rag.models import Chunk

DATASET_NAME = "C-MTEB/T2Reranking"
DATASET_LICENSE = "Apache-2.0 (original THUIR/T2Ranking dataset)"
SOURCE_URL = (
    "https://huggingface.co/datasets/C-MTEB/T2Reranking/resolve/main/"
    "data/dev-00000-of-00001-65d96bde8023d9b9.parquet"
)
DEFAULT_CACHE_PATH = Path("eval/.cache/t2_reranking/dev.parquet")
DEFAULT_OUTPUT_DIR = Path("eval/dataset/t2_reranking")
DEFAULT_QUERY_COUNT = 100
DEFAULT_SEED = 42
DEFAULT_MAX_POSITIVES = 20
DEFAULT_MAX_NEGATIVES = 20

_TAG_PATTERN = re.compile(r"<[^>]+>")
_WHITESPACE_PATTERN = re.compile(r"\s+")


class RetrievalDatasetError(RuntimeError):
    """公开检索数据无法下载、解析或转换。"""


@dataclass(frozen=True)
class RetrievalDatasetReport:
    """一次公开检索子集转换的结构化结果。"""

    output_dir: Path
    query_count: int
    passage_count: int
    positive_relation_count: int
    negative_relation_count: int
    seed: int


@dataclass(frozen=True)
class RetrievalCase:
    """共享语料库中的一个检索查询及其相关 passage。"""

    case_id: str
    query: str
    gold_passage_ids: tuple[str, ...]
    source_candidate_passage_ids: tuple[str, ...]


@dataclass(frozen=True)
class PreparedRetrievalDataset:
    """可以直接交给现有 BM25/向量索引原语的评测数据。"""

    chunks: tuple[Chunk, ...]
    cases: tuple[RetrievalCase, ...]


@dataclass(frozen=True)
class _SelectedRow:
    source_index: int
    query: str
    positives: tuple[str, ...]
    negatives: tuple[str, ...]


def download_source(
    destination: Path | str = DEFAULT_CACHE_PATH,
    *,
    source_url: str = SOURCE_URL,
    overwrite: bool = False,
) -> Path:
    """把公开 Parquet 下载到本地缓存，不执行哈希校验。"""
    path = Path(destination)
    if path.is_file() and path.stat().st_size > 0 and not overwrite:
        return path
    if path.exists() and not path.is_file():
        raise RetrievalDatasetError(f"cache destination is not a file: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.download")
    temporary_path.unlink(missing_ok=True)
    try:
        with requests.get(
            source_url,
            stream=True,
            timeout=(15, 180),
        ) as response:
            response.raise_for_status()
            with temporary_path.open("xb") as output:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        output.write(block)
                output.flush()
                os.fsync(output.fileno())
        if temporary_path.stat().st_size == 0:
            raise RetrievalDatasetError("downloaded dataset file is empty")
        os.replace(temporary_path, path)
    except (OSError, requests.RequestException) as exc:
        raise RetrievalDatasetError(f"failed to download {source_url}: {exc}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def iter_parquet_rows(path: Path | str) -> Iterator[dict[str, Any]]:
    """分批读取 Parquet，避免把完整 120 MB 数据一次性展开到内存。"""
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - 当前锁文件由 Streamlit 引入
        raise RetrievalDatasetError(
            "reading T2Reranking requires pyarrow from the locked project environment"
        ) from exc

    source_path = Path(path)
    if not source_path.is_file():
        raise RetrievalDatasetError(f"dataset source not found: {source_path}")
    try:
        parquet_file = parquet.ParquetFile(source_path)
        for batch in parquet_file.iter_batches(
            batch_size=128,
            columns=["query", "positive", "negative"],
        ):
            yield from batch.to_pylist()
    except Exception as exc:
        raise RetrievalDatasetError(
            f"failed to read T2Reranking parquet: {source_path}"
        ) from exc


def prepare_retrieval_dataset(
    rows: Iterable[Mapping[str, object]],
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    query_count: int = DEFAULT_QUERY_COUNT,
    seed: int = DEFAULT_SEED,
    max_positives: int = DEFAULT_MAX_POSITIVES,
    max_negatives: int = DEFAULT_MAX_NEGATIVES,
    overwrite: bool = False,
) -> RetrievalDatasetReport:
    """固定抽样并生成共享 ``corpus / queries / qrels`` JSONL。"""
    _require_positive_int("query_count", query_count)
    _require_positive_int("max_positives", max_positives)
    _require_positive_int("max_negatives", max_negatives)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    selected = _reservoir_sample(rows, query_count=query_count, seed=seed)
    parsed_rows = tuple(
        _parse_selected_row(
            source_index,
            raw,
            max_positives=max_positives,
            max_negatives=max_negatives,
        )
        for source_index, raw in selected
    )

    passages: list[dict[str, str]] = []
    passage_id_by_text: dict[str, str] = {}
    query_records: list[dict[str, object]] = []
    qrel_records: list[dict[str, object]] = []
    positive_relation_count = 0
    negative_relation_count = 0

    def register_passage(text: str) -> str:
        existing = passage_id_by_text.get(text)
        if existing is not None:
            return existing
        passage_id = f"T2P-{len(passages) + 1:06d}"
        passage_id_by_text[text] = passage_id
        passages.append(
            {
                "id": passage_id,
                "text": text,
                "dataset": DATASET_NAME,
            }
        )
        return passage_id

    for case_number, row in enumerate(parsed_rows, start=1):
        case_id = f"T2R-{case_number:03d}"
        positive_ids = tuple(register_passage(text) for text in row.positives)
        positive_id_set = set(positive_ids)
        negative_ids = tuple(
            passage_id
            for passage_id in (register_passage(text) for text in row.negatives)
            if passage_id not in positive_id_set
        )
        source_candidate_ids = tuple(dict.fromkeys((*positive_ids, *negative_ids)))
        if not negative_ids:
            raise RetrievalDatasetError(
                f"selected query has no negative passage after deduplication: {row.query}"
            )
        query_records.append(
            {
                "id": case_id,
                "query": row.query,
                "gold_passage_ids": list(dict.fromkeys(positive_ids)),
                "source_candidate_passage_ids": list(source_candidate_ids),
                "source_row": row.source_index,
            }
        )
        for passage_id in dict.fromkeys(positive_ids):
            qrel_records.append(
                {
                    "query_id": case_id,
                    "passage_id": passage_id,
                    "relevance": 1,
                }
            )
            positive_relation_count += 1
        negative_relation_count += len(dict.fromkeys(negative_ids))

    output_path = Path(output_dir)
    metadata = {
        "dataset": DATASET_NAME,
        "source_url": SOURCE_URL,
        "license": DATASET_LICENSE,
        "retrieval_unit": "passage",
        "candidate_pool": "shared_across_all_selected_queries",
        "query_count": len(query_records),
        "passage_count": len(passages),
        "positive_relation_count": positive_relation_count,
        "negative_relation_count": negative_relation_count,
        "seed": seed,
        "max_positives_per_query": max_positives,
        "max_negatives_per_query": max_negatives,
    }
    _write_dataset_directory(
        output_path,
        corpus=passages,
        queries=query_records,
        qrels=qrel_records,
        metadata=metadata,
        overwrite=overwrite,
    )
    return RetrievalDatasetReport(
        output_dir=output_path,
        query_count=len(query_records),
        passage_count=len(passages),
        positive_relation_count=positive_relation_count,
        negative_relation_count=negative_relation_count,
        seed=seed,
    )


def load_prepared_dataset(
    directory: Path | str = DEFAULT_OUTPUT_DIR,
) -> PreparedRetrievalDataset:
    """读取转换产物，并把 passage 适配为现有索引可接收的 ``Chunk``。"""
    root = Path(directory)
    corpus = _read_jsonl(root / "corpus.jsonl")
    queries = _read_jsonl(root / "queries.jsonl")
    qrels = _read_jsonl(root / "qrels.jsonl")

    chunks: list[Chunk] = []
    passage_ids: set[str] = set()
    for row_number, row in enumerate(corpus, start=1):
        passage_id = _required_text(row, "id", row_number=row_number)
        text = _required_text(row, "text", row_number=row_number)
        if passage_id in passage_ids:
            raise RetrievalDatasetError(f"duplicate passage id: {passage_id}")
        passage_ids.add(passage_id)
        chunks.append(
            Chunk(
                id=passage_id,
                text=text,
                metadata={
                    "source_path": f"{DATASET_NAME}/{passage_id}",
                    "chunk_index": 0,
                    "benchmark_dataset": DATASET_NAME,
                },
            )
        )

    gold_by_query: dict[str, list[str]] = {}
    for row_number, row in enumerate(qrels, start=1):
        query_id = _required_text(row, "query_id", row_number=row_number)
        passage_id = _required_text(row, "passage_id", row_number=row_number)
        if passage_id not in passage_ids:
            raise RetrievalDatasetError(
                f"qrel references unknown passage id: {passage_id}"
            )
        if row.get("relevance") != 1:
            raise RetrievalDatasetError("prepared qrels must contain relevance=1")
        gold_by_query.setdefault(query_id, []).append(passage_id)

    cases: list[RetrievalCase] = []
    case_ids: set[str] = set()
    for row_number, row in enumerate(queries, start=1):
        case_id = _required_text(row, "id", row_number=row_number)
        query = _required_text(row, "query", row_number=row_number)
        if case_id in case_ids:
            raise RetrievalDatasetError(f"duplicate query id: {case_id}")
        case_ids.add(case_id)
        gold_ids = tuple(dict.fromkeys(gold_by_query.get(case_id, ())))
        if not gold_ids:
            raise RetrievalDatasetError(f"query has no qrels: {case_id}")
        candidate_ids = _required_text_list(
            row,
            "source_candidate_passage_ids",
            row_number=row_number,
        )
        if not set(candidate_ids).issubset(passage_ids):
            raise RetrievalDatasetError(
                f"query references unknown source candidate: {case_id}"
            )
        cases.append(
            RetrievalCase(
                case_id=case_id,
                query=query,
                gold_passage_ids=gold_ids,
                source_candidate_passage_ids=candidate_ids,
            )
        )
    if set(gold_by_query) != case_ids:
        raise RetrievalDatasetError("qrels and queries contain different query ids")
    return PreparedRetrievalDataset(chunks=tuple(chunks), cases=tuple(cases))


def _reservoir_sample(
    rows: Iterable[Mapping[str, object]],
    *,
    query_count: int,
    seed: int,
) -> list[tuple[int, Mapping[str, object]]]:
    rng = random.Random(seed)
    selected: list[tuple[int, Mapping[str, object]]] = []
    eligible_count = 0
    seen_queries: set[str] = set()
    for source_index, row in enumerate(rows):
        query = _source_query(row)
        if query is None or query in seen_queries:
            continue
        if not _has_text_sequence(row.get("positive")):
            continue
        if not _has_text_sequence(row.get("negative")):
            continue
        seen_queries.add(query)
        eligible_count += 1
        item = (source_index, row)
        if len(selected) < query_count:
            selected.append(item)
            continue
        replacement = rng.randrange(eligible_count)
        if replacement < query_count:
            selected[replacement] = item
    if eligible_count < query_count:
        raise RetrievalDatasetError(
            f"requested {query_count} queries but only {eligible_count} valid rows exist"
        )
    return sorted(selected, key=lambda item: item[0])


def _parse_selected_row(
    source_index: int,
    row: Mapping[str, object],
    *,
    max_positives: int,
    max_negatives: int,
) -> _SelectedRow:
    query = _clean_text(str(row["query"]))
    positives = _clean_sequence(row["positive"], limit=max_positives)
    positive_set = set(positives)
    negatives = tuple(
        value
        for value in _clean_sequence(row["negative"], limit=max_negatives)
        if value not in positive_set
    )
    if not query or not positives or not negatives:
        raise RetrievalDatasetError(f"invalid selected source row: {source_index}")
    return _SelectedRow(
        source_index=source_index,
        query=query,
        positives=positives,
        negatives=negatives,
    )


def _source_query(row: Mapping[str, object]) -> str | None:
    value = row.get("query")
    if not isinstance(value, str):
        return None
    normalized = _clean_text(value)
    return normalized or None


def _has_text_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and any(
        isinstance(item, str) and item.strip() for item in value
    )


def _clean_sequence(value: object, *, limit: int) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        text = _clean_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return tuple(cleaned)


def _clean_text(value: str) -> str:
    without_tags = _TAG_PATTERN.sub(" ", html.unescape(value))
    return _WHITESPACE_PATTERN.sub(" ", without_tags).strip()


def _write_dataset_directory(
    output_path: Path,
    *,
    corpus: Sequence[Mapping[str, object]],
    queries: Sequence[Mapping[str, object]],
    qrels: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object],
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output directory already exists: {output_path}; pass overwrite=True"
        )
    if output_path.exists():
        _require_owned_output_directory(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.parent / f".{output_path.name}-{uuid.uuid4().hex}"
    temporary_path.mkdir()
    try:
        _write_jsonl(temporary_path / "corpus.jsonl", corpus)
        _write_jsonl(temporary_path / "queries.jsonl", queries)
        _write_jsonl(temporary_path / "qrels.jsonl", qrels)
        (temporary_path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if output_path.exists():
            shutil.rmtree(output_path)
        temporary_path.replace(output_path)
    finally:
        shutil.rmtree(temporary_path, ignore_errors=True)


def _require_owned_output_directory(output_path: Path) -> None:
    """只允许覆盖本转换器此前生成的目录，避免误删任意路径。"""
    metadata_path = output_path / "metadata.json"
    if not output_path.is_dir() or not metadata_path.is_file():
        raise RetrievalDatasetError(
            f"refusing to overwrite an unowned output directory: {output_path}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalDatasetError(
            f"refusing to overwrite output with invalid metadata: {output_path}"
        ) from exc
    if not isinstance(metadata, dict) or metadata.get("dataset") != DATASET_NAME:
        raise RetrievalDatasetError(
            f"refusing to overwrite output owned by another dataset: {output_path}"
        )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise RetrievalDatasetError(f"prepared dataset file not found: {path}")
    rows: list[dict[str, object]] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RetrievalDatasetError(
                        f"JSONL row must be an object: {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalDatasetError(f"failed to read prepared dataset: {path}") from exc
    if not rows:
        raise RetrievalDatasetError(f"prepared dataset file is empty: {path}")
    return rows


def _required_text(
    row: Mapping[str, object],
    field: str,
    *,
    row_number: int,
) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RetrievalDatasetError(
            f"row {row_number} field {field!r} must be non-empty text"
        )
    return value.strip()


def _required_text_list(
    row: Mapping[str, object],
    field: str,
    *,
    row_number: int,
) -> tuple[str, ...]:
    value = row.get(field)
    if not isinstance(value, list) or not value:
        raise RetrievalDatasetError(
            f"row {row_number} field {field!r} must be a non-empty list"
        )
    normalized = tuple(
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    )
    if len(normalized) != len(value):
        raise RetrievalDatasetError(
            f"row {row_number} field {field!r} must contain only non-empty strings"
        )
    return normalized


def _require_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "DATASET_LICENSE",
    "DATASET_NAME",
    "DEFAULT_CACHE_PATH",
    "DEFAULT_MAX_NEGATIVES",
    "DEFAULT_MAX_POSITIVES",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_QUERY_COUNT",
    "DEFAULT_SEED",
    "PreparedRetrievalDataset",
    "RetrievalCase",
    "RetrievalDatasetError",
    "RetrievalDatasetReport",
    "SOURCE_URL",
    "download_source",
    "iter_parquet_rows",
    "load_prepared_dataset",
    "prepare_retrieval_dataset",
]
