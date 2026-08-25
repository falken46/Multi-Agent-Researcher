from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.retrieval_dataset import (
    RetrievalDatasetError,
    load_prepared_dataset,
    prepare_retrieval_dataset,
)


def test_prepare_builds_one_shared_pool_with_query_specific_qrels(
    runtime_dir: Path,
) -> None:
    rows = [
        {
            "query": "如何取快递？",
            "positive": ["<p>使用取件码</p>"],
            "negative": ["联系快递员", "公共干扰段落"],
        },
        {
            "query": "如何联系快递员？",
            "positive": ["联系快递员"],
            "negative": ["使用取件码", "另一条干扰段落"],
        },
    ]
    output_dir = runtime_dir / "prepared"

    report = prepare_retrieval_dataset(
        rows,
        output_dir,
        query_count=2,
        seed=42,
    )
    dataset = load_prepared_dataset(output_dir)

    assert report.query_count == 2
    assert report.passage_count == 4
    assert report.positive_relation_count == 2
    assert report.negative_relation_count == 4
    assert len(dataset.chunks) == 4
    assert len(dataset.cases) == 2
    assert all(chunk.metadata["chunk_index"] == 0 for chunk in dataset.chunks)
    assert "<p>" not in dataset.chunks[0].text

    passage_id_by_text = {chunk.text: chunk.id for chunk in dataset.chunks}
    first, second = dataset.cases
    assert first.gold_passage_ids == (passage_id_by_text["使用取件码"],)
    assert second.gold_passage_ids == (passage_id_by_text["联系快递员"],)
    assert passage_id_by_text["联系快递员"] in first.source_candidate_passage_ids
    assert passage_id_by_text["使用取件码"] in second.source_candidate_passage_ids


def test_fixed_seed_produces_identical_prepared_files(runtime_dir: Path) -> None:
    rows = [
        {
            "query": f"问题 {index}",
            "positive": [f"正例 {index}"],
            "negative": [f"负例 {index}"],
        }
        for index in range(12)
    ]
    first_dir = runtime_dir / "first"
    second_dir = runtime_dir / "second"

    prepare_retrieval_dataset(rows, first_dir, query_count=5, seed=7)
    prepare_retrieval_dataset(rows, second_dir, query_count=5, seed=7)

    for filename in ("corpus.jsonl", "queries.jsonl", "qrels.jsonl", "metadata.json"):
        assert (first_dir / filename).read_bytes() == (second_dir / filename).read_bytes()


def test_prepare_rejects_too_few_valid_rows_and_existing_output(
    runtime_dir: Path,
) -> None:
    rows = [
        {"query": "有效问题", "positive": ["正例"], "negative": ["负例"]},
        {"query": "无负例", "positive": ["正例"], "negative": []},
    ]

    with pytest.raises(RetrievalDatasetError, match="only 1 valid rows"):
        prepare_retrieval_dataset(
            rows,
            runtime_dir / "too-many",
            query_count=2,
            seed=1,
        )

    output_dir = runtime_dir / "existing"
    prepare_retrieval_dataset(rows[:1], output_dir, query_count=1, seed=1)
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_retrieval_dataset(rows[:1], output_dir, query_count=1, seed=1)

    prepare_retrieval_dataset(
        rows[:1],
        output_dir,
        query_count=1,
        seed=1,
        overwrite=True,
    )

    unowned_dir = runtime_dir / "unowned"
    unowned_dir.mkdir()
    (unowned_dir / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(RetrievalDatasetError, match="unowned"):
        prepare_retrieval_dataset(
            rows[:1],
            unowned_dir,
            query_count=1,
            seed=1,
            overwrite=True,
        )
    assert (unowned_dir / "keep.txt").read_text(encoding="utf-8") == "user data"


def test_loader_fails_closed_when_qrels_reference_unknown_passage(
    runtime_dir: Path,
) -> None:
    output_dir = runtime_dir / "prepared"
    prepare_retrieval_dataset(
        [{"query": "问题", "positive": ["正例"], "negative": ["负例"]}],
        output_dir,
        query_count=1,
        seed=1,
    )
    qrels_path = output_dir / "qrels.jsonl"
    qrel = json.loads(qrels_path.read_text(encoding="utf-8"))
    qrel["passage_id"] = "UNKNOWN"
    qrels_path.write_text(
        json.dumps(qrel, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RetrievalDatasetError, match="unknown passage"):
        load_prepared_dataset(output_dir)
