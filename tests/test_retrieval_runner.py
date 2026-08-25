from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import eval.retrieval_runner as runner_module
from core.config import Settings
from eval.models import RetrievalObservation
from eval.prepare_retrieval_dataset import prepare_retrieval_dataset
from eval.retrieval_runner import run_retrieval_evaluation
from rag.embeddings import FakeEmbeddingBackend
from rag.models import RetrievalResult


class CountingFakeEmbeddingBackend(FakeEmbeddingBackend):
    def __init__(self) -> None:
        super().__init__(dimension=64)
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return super().embed_query(text)


def test_runner_reuses_indexes_and_r4_reranks_the_same_r3_candidates(
    runtime_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = _prepared_dataset(runtime_dir)
    output_path = runtime_dir / "raw" / "retrieval.jsonl"
    embedder = CountingFakeEmbeddingBackend()

    def fake_rerank(
        query: str,
        candidates: list[RetrievalResult],
        *,
        top_n: int,
        settings: Settings,
    ) -> list[RetrievalResult]:
        del query, settings
        return list(reversed(candidates))[:top_n]

    monkeypatch.setattr(runner_module, "rerank", fake_rerank)
    settings = Settings(
        _env_file=None,
        embedding_backend="fake",
        rerank_backend="onnx",
    )

    report = run_retrieval_evaluation(
        dataset_dir=dataset_dir,
        output_path=output_path,
        index_dir=runtime_dir / "indexes",
        settings=settings,
        embedding_backend=embedder,
    )
    records = _read_jsonl(output_path)

    assert report.query_count == 2
    assert report.observation_count == 8
    assert report.groups == ("R1", "R2", "R3", "R4")
    assert embedder.document_calls == 1
    assert embedder.query_calls == 2
    assert len(report.summaries) == 4
    assert all(isinstance(RetrievalObservation.from_raw(item), RetrievalObservation) for item in records)

    by_case_group = {
        (record["case_id"], record["group"]): record for record in records
    }
    for case_id in ("T2R-001", "T2R-002"):
        r3 = by_case_group[(case_id, "R3")]
        r4 = by_case_group[(case_id, "R4")]
        assert r4["candidate_chunk_ids"] == r3["candidate_chunk_ids"]
        assert r4["retrieved_chunk_ids"] == list(
            reversed(r3["candidate_chunk_ids"])
        )[:5]


def test_runner_rejects_network_embedding_and_non_onnx_r4(
    runtime_dir: Path,
) -> None:
    dataset_dir = _prepared_dataset(runtime_dir)

    with pytest.raises(ValueError, match="local fastembed or fake"):
        run_retrieval_evaluation(
            dataset_dir=dataset_dir,
            output_path=runtime_dir / "remote.jsonl",
            index_dir=runtime_dir / "remote-index",
            groups=("R1",),
            settings=Settings(
                _env_file=None,
                embedding_backend="remote",
                embedding_remote_url="https://example.invalid/embeddings",
            ),
        )

    with pytest.raises(ValueError, match="requires rerank_backend='onnx'"):
        run_retrieval_evaluation(
            dataset_dir=dataset_dir,
            output_path=runtime_dir / "llm.jsonl",
            index_dir=runtime_dir / "llm-index",
            groups=("R4",),
            settings=Settings(
                _env_file=None,
                embedding_backend="fake",
                rerank_backend="llm",
            ),
        )


def test_runner_refuses_to_overwrite_raw_evidence(runtime_dir: Path) -> None:
    dataset_dir = _prepared_dataset(runtime_dir)
    output_path = runtime_dir / "existing.jsonl"
    output_path.write_text("keep\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        run_retrieval_evaluation(
            dataset_dir=dataset_dir,
            output_path=output_path,
            index_dir=runtime_dir / "indexes",
            groups=("R2",),
            settings=Settings(_env_file=None),
        )
    assert output_path.read_text(encoding="utf-8") == "keep\n"


def test_retrieval_cli_runs_bm25_only(
    runtime_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from eval.run_retrieval import main

    dataset_dir = _prepared_dataset(runtime_dir)
    output_path = runtime_dir / "cli.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_retrieval",
            "--dataset",
            str(dataset_dir),
            "--index-dir",
            str(runtime_dir / "cli-index"),
            "--output",
            str(output_path),
            "--groups",
            "R2",
        ],
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["groups"] == ["R2"]
    assert payload["query_count"] == 2
    assert len(_read_jsonl(output_path)) == 2


def _prepared_dataset(runtime_dir: Path) -> Path:
    output_dir = runtime_dir / "prepared"
    prepare_retrieval_dataset(
        [
            {
                "query": "苹果手机如何充电",
                "positive": ["苹果手机使用充电线补充电量"],
                "negative": ["香蕉是黄色水果", "电脑需要键盘输入"],
            },
            {
                "query": "香蕉是什么颜色",
                "positive": ["成熟香蕉的外皮通常是黄色"],
                "negative": ["苹果手机需要充电", "键盘属于电脑外设"],
            },
        ],
        output_dir,
        query_count=2,
        seed=42,
    )
    return output_dir


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
