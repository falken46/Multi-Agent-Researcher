from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import eval.orchestration_runner as runner_module
from agents.graph import create_initial_state
from agents.state import ResearchState
from core.config import Settings
from eval.models import TaskObservation
from eval.orchestration_runner import (
    ExperimentGroup,
    OrchestrationRunReport,
    PlannedTask,
    TaskExecution,
    load_orchestration_cases,
    prewarm_initial_query_cache,
    run_orchestration_evaluation,
)
from eval.query_cache import QueryCache
from tools.web_search import SearchResult


def test_default_dataset_contains_balanced_frozen_cases() -> None:
    cases = load_orchestration_cases()

    assert len(cases) == 15
    assert {case.case_id for case in cases} == {
        *(f"O-KB-{index:03d}" for index in range(1, 6)),
        *(f"O-WEB-{index:03d}" for index in range(1, 6)),
        *(f"O-MIXED-{index:03d}" for index in range(1, 6)),
    }
    assert sum(case.case_type == "kb" for case in cases) == 5
    assert sum(case.case_type == "web" for case in cases) == 5
    assert sum(case.case_type == "mixed" for case in cases) == 5
    assert all(len(case.sub_questions) >= 3 for case in cases)
    assert all(case.must_cover for case in cases)


@pytest.mark.asyncio
async def test_runner_plans_p_only_for_kb_and_q_for_all_cases_then_resumes(
    runtime_dir: Path,
) -> None:
    dataset_path = _write_dataset(runtime_dir)
    output_path = runtime_dir / "raw" / "tasks.jsonl"
    calls: list[tuple[str, str, int, int, bool]] = []

    async def fake_executor(
        task: PlannedTask,
        *,
        cache: QueryCache,
        settings: Settings,
    ) -> TaskExecution:
        del cache, settings
        calls.append(
            (
                task.case.case_id,
                task.group.name,
                task.round_index,
                task.group.concurrency,
                task.group.critic_enabled,
            )
        )
        return _fake_execution(task)

    settings = Settings(
        _env_file=None,
        trace_enabled=False,
        research_concurrency=3,
    )
    report = await run_orchestration_evaluation(
        output_path=output_path,
        snapshot_id="snapshot-1",
        cache_mode="replay-only",
        max_tasks=8,
        dataset_path=dataset_path,
        cache_dir=runtime_dir / "cache",
        settings=settings,
        executor=fake_executor,
    )
    records = _read_jsonl(output_path)

    assert report.planned_task_count == 8
    assert report.executed_task_count == 8
    assert report.skipped_task_count == 0
    assert [item[1] for item in calls[:4]] == ["P1", "P2", "Q1", "Q2"]
    assert calls[0][3:] == (1, False)
    assert calls[1][3:] == (3, False)
    assert calls[3][3:] == (3, True)
    assert len(records) == 8
    assert all(isinstance(TaskObservation.from_raw(item), TaskObservation) for item in records)

    resumed = await run_orchestration_evaluation(
        output_path=output_path,
        snapshot_id="snapshot-1",
        cache_mode="replay-only",
        max_tasks=1,
        dataset_path=dataset_path,
        cache_dir=runtime_dir / "cache",
        settings=settings,
        executor=fake_executor,
        resume=True,
    )
    assert resumed.executed_task_count == 0
    assert resumed.skipped_task_count == 8
    assert len(calls) == 8

    with pytest.raises(ValueError, match="cache_mode"):
        await run_orchestration_evaluation(
            output_path=output_path,
            snapshot_id="snapshot-1",
            cache_mode="record",
            max_tasks=1,
            dataset_path=dataset_path,
            cache_dir=runtime_dir / "cache",
            settings=settings,
            executor=fake_executor,
            resume=True,
        )


@pytest.mark.asyncio
async def test_runner_hard_limit_and_live_gate_fail_before_execution(
    runtime_dir: Path,
) -> None:
    dataset_path = _write_dataset(runtime_dir)
    calls = 0

    async def fake_executor(
        task: PlannedTask,
        *,
        cache: QueryCache,
        settings: Settings,
    ) -> TaskExecution:
        nonlocal calls
        del cache, settings
        calls += 1
        return _fake_execution(task)

    with pytest.raises(ValueError, match="max_tasks=7"):
        await run_orchestration_evaluation(
            output_path=runtime_dir / "limited.jsonl",
            snapshot_id="snapshot-1",
            cache_mode="replay-only",
            max_tasks=7,
            dataset_path=dataset_path,
            settings=Settings(_env_file=None, research_concurrency=3),
            executor=fake_executor,
        )
    assert calls == 0
    assert not (runtime_dir / "limited.jsonl").exists()

    with pytest.raises(ValueError, match="explicit live=True"):
        await run_orchestration_evaluation(
            output_path=runtime_dir / "not-live.jsonl",
            snapshot_id="snapshot-1",
            cache_mode="replay-only",
            max_tasks=1,
            dataset_path=dataset_path,
            groups=("P1",),
            case_limit=1,
            settings=Settings(_env_file=None, trace_enabled=True),
        )


def test_prewarm_fetches_each_frozen_query_once_then_reuses_cache(
    runtime_dir: Path,
) -> None:
    dataset_path = _write_dataset(runtime_dir)
    calls: list[str] = []

    def fake_fetcher(
        query: str,
        max_results: int = 3,
        provider: str | None = None,
    ) -> list[SearchResult]:
        assert max_results == 3
        assert provider == "tavily"
        calls.append(query)
        return [_search_result(query)]

    kwargs = {
        "snapshot_id": "snapshot-1",
        "max_queries": 6,
        "dataset_path": dataset_path,
        "cache_dir": runtime_dir / "cache",
        "settings": Settings(_env_file=None, search_provider="tavily"),
        "fetcher": fake_fetcher,
    }
    first = prewarm_initial_query_cache(**kwargs)
    second = prewarm_initial_query_cache(**kwargs)

    assert first.query_count == 6
    assert first.fetched_count == 6
    assert first.cached_count == 0
    assert second.fetched_count == 0
    assert second.cached_count == 6
    assert len(calls) == 6


@pytest.mark.asyncio
async def test_graph_executor_uses_fixed_planner_cache_and_critic_switch(
    runtime_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = load_orchestration_cases(_write_dataset(runtime_dir))[0]
    critic_calls = 0

    async def fake_researcher(
        state: ResearchState,
        *,
        writer: object,
        settings: Settings,
        web_searcher: object,
        fail_fast_exceptions: tuple[type[BaseException], ...],
    ) -> dict[str, object]:
        del writer
        assert settings.research_concurrency in {1, 3}
        assert fail_fast_exceptions
        search = web_searcher
        assert callable(search)
        citations = {}
        results = {}
        for question in state["sub_questions"]:
            web_results = search(question, max_results=3)
            results[question] = web_results[0]["snippet"]
            citations[question] = [
                {
                    "source": web_results[0]["url"],
                    "origin": "web",
                    "snippet": web_results[0]["snippet"],
                }
            ]
        return {"research_results": results, "citations": citations, "errors": []}

    async def fake_critic(state: ResearchState) -> dict[str, object]:
        nonlocal critic_calls
        critic_calls += 1
        return {
            "quality_score": 0.9,
            "quality_history": [0.9],
            "critique": "enough",
            "missing_aspects": [],
        }

    def fake_writer(state: ResearchState) -> dict[str, object]:
        assert state["sub_questions"] == list(case.sub_questions)
        return {"final_report": f"{case.must_cover[0][0]} [1]"}

    def fake_web_search(
        query: str,
        max_results: int = 3,
        provider: str | None = None,
    ) -> list[SearchResult]:
        del max_results, provider
        return [_search_result(query)]

    monkeypatch.setattr(runner_module, "researcher_node", fake_researcher)
    monkeypatch.setattr(runner_module, "critic_node", fake_critic)
    monkeypatch.setattr(runner_module, "writer_node", fake_writer)
    monkeypatch.setattr(runner_module, "web_search", fake_web_search)
    settings = Settings(
        _env_file=None,
        trace_enabled=False,
        research_concurrency=3,
    )
    cache = QueryCache(runtime_dir / "cache", snapshot_id="snapshot-1", mode="record")

    p1 = PlannedTask(case, ExperimentGroup("P1", "P", 1, False), 1)
    p1_execution = await runner_module._execute_graph_task(
        p1,
        cache=cache,
        settings=settings,
    )
    q2 = PlannedTask(case, ExperimentGroup("Q2", "Q", 3, True), 1)
    q2_execution = await runner_module._execute_graph_task(
        q2,
        cache=cache,
        settings=settings,
    )

    assert p1_execution.final_state["sub_questions"] == list(case.sub_questions)
    assert q2_execution.final_state["sub_questions"] == list(case.sub_questions)
    assert critic_calls == 1
    assert p1_execution.final_state["critique"].startswith("Critic disabled")
    assert q2_execution.final_state["quality_score"] == pytest.approx(0.9)


def test_orchestration_cli_forwards_safety_arguments(
    runtime_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import eval.run_orchestration as cli_module

    captured: dict[str, object] = {}

    async def fake_run(**kwargs: object) -> OrchestrationRunReport:
        captured.update(kwargs)
        return OrchestrationRunReport(
            output_path=Path(str(kwargs["output_path"])),
            planned_task_count=2,
            executed_task_count=2,
            skipped_task_count=0,
            groups=("P1", "P2"),
            rounds=1,
        )

    monkeypatch.setattr(cli_module, "run_orchestration_evaluation", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_orchestration",
            "--snapshot-id",
            "snapshot-1",
            "--cache-mode",
            "replay-only",
            "--groups",
            "P1",
            "P2",
            "--max-tasks",
            "2",
            "--output",
            str(runtime_dir / "raw.jsonl"),
            "--live",
        ],
    )

    cli_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["executed_task_count"] == 2
    assert captured["live"] is True
    assert captured["max_tasks"] == 2
    assert captured["cache_mode"] == "replay-only"


def _fake_execution(task: PlannedTask) -> TaskExecution:
    state = create_initial_state(task.case.topic, trace_id=f"trace-{task.case.case_id}")
    state["sub_questions"] = list(task.case.sub_questions)
    state["final_report"] = f"{task.case.must_cover[0][0]} [1]"
    state["citations"] = {
        task.case.sub_questions[0]: [
            {
                "source": "https://example.com/source",
                "origin": "web",
                "snippet": "evidence",
            }
        ]
    }
    state["quality_score"] = 0.8 if task.group.critic_enabled else 1.0
    state["quality_history"] = [0.8] if task.group.critic_enabled else []
    return TaskExecution(
        final_state=state,
        trace_summary={
            "total_latency_ms": 100.0,
            "total_tokens": 10,
            "total_cost": 0.001,
            "llm_calls": 1,
            "fallback_count": 1,
            "revision_count": int(task.group.critic_enabled),
            "currency": "CNY",
            "pricing_version": "test-v1",
            "by_node": {"researcher": {"latency_ms": 60.0}},
        },
    )


def _write_dataset(runtime_dir: Path) -> Path:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    path = runtime_dir / "orchestration.jsonl"
    records = []
    for prefix, case_type in (("KB", "kb"), ("WEB", "web"), ("MIX", "mixed")):
        records.append(
            {
                "id": f"O-{prefix}-001",
                "type": case_type,
                "topic": f"{case_type} topic",
                "sub_questions": [f"{case_type} q{index}" for index in range(1, 3)],
                "must_cover": [
                    {"id": f"{case_type}-concept", "any_of": [f"{case_type} answer"]}
                ],
            }
        )
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    return path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _search_result(query: str) -> SearchResult:
    return {
        "title": query,
        "url": f"https://example.com/{query}",
        "snippet": f"{query} evidence",
        "source": "tavily",
    }
