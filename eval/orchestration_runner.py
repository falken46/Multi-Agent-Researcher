"""Phase 13 P/Q 轨端到端编排评测 runner。

P1/P2 只改变 Researcher 并发上限；Q1/Q2 只改变 Critic 是否启用。
所有组使用数据集冻结的 Planner 子问题，并通过 QueryCache 控制 Web 原始证据。
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from langgraph.types import StreamWriter

from agents.critic import critic_node
from agents.graph import build_graph, create_initial_state
from agents.researcher import researcher_node
from agents.state import Citation, ResearchState
from agents.writer import writer_node
from core.config import Settings, get_settings
from core.trace import emit, new_trace_id, read_events, summarize
from eval.models import CoverConcept, TaskObservation, TaskTrack
from eval.query_cache import CacheMode, QueryCache, QueryCacheError, SearchFetcher
from tools.web_search import SearchProvider, SearchResult, web_search

OrchestrationGroup = Literal["P1", "P2", "Q1", "Q2"]
CaseType = Literal["kb", "web", "mixed"]

DEFAULT_DATASET_PATH = Path("eval/dataset/orchestration_qa.jsonl")
DEFAULT_WEB_CACHE_DIR = Path("eval/.cache/web")
DEFAULT_RAW_DIR = Path("eval/reports/raw")
DEFAULT_GROUPS: tuple[OrchestrationGroup, ...] = ("P1", "P2", "Q1", "Q2")


class OrchestrationDatasetError(RuntimeError):
    """编排评测题集不符合冻结 schema。"""


@dataclass(frozen=True)
class OrchestrationCase:
    case_id: str
    case_type: CaseType
    topic: str
    sub_questions: tuple[str, ...]
    must_cover_ids: tuple[str, ...]
    must_cover: tuple[CoverConcept, ...]


@dataclass(frozen=True)
class ExperimentGroup:
    name: OrchestrationGroup
    track: TaskTrack
    concurrency: int
    critic_enabled: bool


@dataclass(frozen=True)
class PlannedTask:
    case: OrchestrationCase
    group: ExperimentGroup
    round_index: int

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.case.case_id, self.group.name, self.round_index)


@dataclass(frozen=True)
class TaskExecution:
    final_state: ResearchState
    trace_summary: Mapping[str, Any]
    cache_events: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class OrchestrationRunReport:
    output_path: Path
    planned_task_count: int
    executed_task_count: int
    skipped_task_count: int
    groups: tuple[OrchestrationGroup, ...]
    rounds: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path.as_posix(),
            "planned_task_count": self.planned_task_count,
            "executed_task_count": self.executed_task_count,
            "skipped_task_count": self.skipped_task_count,
            "groups": list(self.groups),
            "rounds": self.rounds,
        }


@dataclass(frozen=True)
class CacheWarmupReport:
    snapshot_id: str
    query_count: int
    fetched_count: int
    cached_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "query_count": self.query_count,
            "fetched_count": self.fetched_count,
            "cached_count": self.cached_count,
        }


class TaskExecutor(Protocol):
    async def __call__(
        self,
        task: PlannedTask,
        *,
        cache: QueryCache,
        settings: Settings,
    ) -> TaskExecution: ...


def default_raw_output_path() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_RAW_DIR / f"orchestration_{timestamp}.jsonl"


def load_orchestration_cases(
    path: Path | str = DEFAULT_DATASET_PATH,
) -> tuple[OrchestrationCase, ...]:
    """读取 5 KB / 5 Web / 5 mixed 冻结题集。"""

    source = Path(path)
    if not source.is_file():
        raise OrchestrationDatasetError(f"orchestration dataset not found: {source}")
    cases: list[OrchestrationCase] = []
    case_ids: set[str] = set()
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise OrchestrationDatasetError(
                f"blank orchestration dataset line: {source}:{line_number}"
            )
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OrchestrationDatasetError(
                f"invalid orchestration JSON: {source}:{line_number}"
            ) from exc
        if not isinstance(raw, Mapping):
            raise OrchestrationDatasetError(
                f"orchestration row must be an object: {source}:{line_number}"
            )
        case = _parse_case(raw, line_number=line_number)
        if case.case_id in case_ids:
            raise OrchestrationDatasetError(f"duplicate case id: {case.case_id}")
        case_ids.add(case.case_id)
        cases.append(case)
    if not cases:
        raise OrchestrationDatasetError("orchestration dataset must not be empty")
    return tuple(cases)


async def run_orchestration_evaluation(
    *,
    output_path: Path | str,
    snapshot_id: str,
    cache_mode: CacheMode,
    max_tasks: int,
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    cache_dir: Path | str = DEFAULT_WEB_CACHE_DIR,
    groups: Sequence[str] = DEFAULT_GROUPS,
    rounds: int = 1,
    case_ids: Sequence[str] | None = None,
    case_limit: int | None = None,
    resume: bool = False,
    live: bool = False,
    settings: Settings | None = None,
    executor: TaskExecutor | None = None,
) -> OrchestrationRunReport:
    """执行受控 P/Q 任务；每完成一题立即追加并 fsync raw JSONL。"""

    _require_positive_int("max_tasks", max_tasks)
    _require_positive_int("rounds", rounds)
    normalized_groups = _normalize_groups(groups)
    current = settings or get_settings()
    if current.research_concurrency <= 1 and any(
        group in {"P2", "Q1", "Q2"} for group in normalized_groups
    ):
        raise ValueError("parallel groups require research_concurrency greater than one")
    if executor is None:
        _validate_live_execution(live=live, cache_mode=cache_mode, settings=current)
        active_executor: TaskExecutor = _execute_graph_task
    else:
        active_executor = executor

    cases = _select_cases(
        load_orchestration_cases(dataset_path),
        case_ids=case_ids,
        case_limit=case_limit,
    )
    group_specs = _group_specs(normalized_groups, current)
    planned = _plan_tasks(cases, groups=group_specs, rounds=rounds)
    if not planned:
        raise ValueError("selected cases and groups produce no orchestration tasks")

    output = Path(output_path)
    existing = _load_existing_records(output, resume=resume)
    completed_keys = {
        _record_key(record): record
        for record in existing
    }
    remaining: list[PlannedTask] = []
    for task in planned:
        existing_record = completed_keys.get(task.key)
        if existing_record is None:
            remaining.append(task)
            continue
        _validate_resume_identity(
            existing_record,
            task=task,
            snapshot_id=snapshot_id,
            cache_mode=cache_mode,
            settings=current,
        )
    if len(remaining) > max_tasks:
        raise ValueError(
            f"planned execution requires {len(remaining)} tasks but max_tasks={max_tasks}"
        )

    cache = QueryCache(cache_dir, snapshot_id=snapshot_id, mode=cache_mode)
    executed_count = 0
    for task in remaining:
        execution = await active_executor(task, cache=cache, settings=current)
        record = _task_record(
            task,
            execution=execution,
            snapshot_id=snapshot_id,
            cache_mode=cache_mode,
            settings=current,
        )
        TaskObservation.from_raw(record)
        _append_jsonl_record(output, record)
        executed_count += 1

    return OrchestrationRunReport(
        output_path=output,
        planned_task_count=len(planned),
        executed_task_count=executed_count,
        skipped_task_count=len(planned) - executed_count,
        groups=normalized_groups,
        rounds=rounds,
    )


def prewarm_initial_query_cache(
    *,
    snapshot_id: str,
    max_queries: int,
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    cache_dir: Path | str = DEFAULT_WEB_CACHE_DIR,
    case_ids: Sequence[str] | None = None,
    case_limit: int | None = None,
    settings: Settings | None = None,
    live: bool = False,
    fetcher: SearchFetcher | None = None,
) -> CacheWarmupReport:
    """预热冻结 Planner 的初始 query；不运行任何 Agent 或 LLM。"""

    _require_positive_int("max_queries", max_queries)
    current = settings or get_settings()
    active_fetcher = fetcher
    if active_fetcher is None:
        if not live:
            raise ValueError("real query-cache warmup requires explicit live=True")
        if current.search_provider == "tavily":
            current.require_tavily_api_key()
        active_fetcher = web_search
    cases = _select_cases(
        load_orchestration_cases(dataset_path),
        case_ids=case_ids,
        case_limit=case_limit,
    )
    queries = tuple(
        dict.fromkeys(
            question
            for case in cases
            for question in case.sub_questions
        )
    )
    if len(queries) > max_queries:
        raise ValueError(
            f"warmup requires {len(queries)} queries but max_queries={max_queries}"
        )
    cache = QueryCache(cache_dir, snapshot_id=snapshot_id, mode="record")
    fetched_count = 0
    cached_count = 0
    for query in queries:
        resolved = cache.resolve(
            query,
            provider=current.search_provider,
            max_results=3,
            fetcher=active_fetcher,
        )
        if resolved.from_cache:
            cached_count += 1
        else:
            fetched_count += 1
    return CacheWarmupReport(
        snapshot_id=snapshot_id,
        query_count=len(queries),
        fetched_count=fetched_count,
        cached_count=cached_count,
    )


async def _execute_graph_task(
    task: PlannedTask,
    *,
    cache: QueryCache,
    settings: Settings,
) -> TaskExecution:
    trace_id = new_trace_id()
    initial_state = create_initial_state(task.case.topic, trace_id=trace_id)
    task_settings = settings.model_copy(
        update={"research_concurrency": task.group.concurrency}
    )
    cached_searcher = _cached_web_searcher(
        cache=cache,
        settings=task_settings,
        trace_id=trace_id,
    )

    async def experiment_researcher(
        state: ResearchState,
        *,
        writer: StreamWriter = _no_op_stream_writer,
    ) -> dict[str, object]:
        return await researcher_node(
            state,
            writer=writer,
            settings=task_settings,
            web_searcher=cached_searcher,
            fail_fast_exceptions=(QueryCacheError,),
        )

    graph = build_graph(
        planner=_fixed_planner_node(task.case),
        researcher=experiment_researcher,
        critic=critic_node if task.group.critic_enabled else _disabled_critic_node,
        writer=writer_node,
    )
    emit(
        {
            "trace_id": trace_id,
            "event": "task_start",
            "payload": {
                "topic": task.case.topic,
                "case_id": task.case.case_id,
                "group": task.group.name,
                "round_index": task.round_index,
            },
        }
    )
    try:
        final_raw = await graph.ainvoke(
            initial_state,
            {"configurable": {"thread_id": trace_id}},
        )
        final_state = cast(ResearchState, final_raw)
        status = "completed" if final_state.get("final_report", "").strip() else "failed"
        emit(
            {
                "trace_id": trace_id,
                "event": "task_end",
                "payload": {
                    "status": status,
                    "case_id": task.case.case_id,
                    "group": task.group.name,
                },
            }
        )
    except Exception as exc:
        emit(
            {
                "trace_id": trace_id,
                "event": "error",
                "node": "eval_runner",
                "payload": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        emit(
            {
                "trace_id": trace_id,
                "event": "task_end",
                "payload": {
                    "status": "failed",
                    "case_id": task.case.case_id,
                    "group": task.group.name,
                },
            }
        )
        raise

    trace_summary = summarize(trace_id)
    cache_events = tuple(
        event
        for event in read_events(trace_id)
        if event.get("event") == "web_cache"
    )
    return TaskExecution(
        final_state=final_state,
        trace_summary=trace_summary,
        cache_events=cache_events,
    )


def _fixed_planner_node(case: OrchestrationCase):
    def fixed_planner(state: ResearchState) -> dict[str, object]:
        trace_id = state["trace_id"]
        started = time.perf_counter()
        emit(
            {
                "trace_id": trace_id,
                "event": "node_start",
                "node": "planner",
                "payload": {
                    "fixed": True,
                    "sub_question_count": len(case.sub_questions),
                },
            }
        )
        result = {"sub_questions": list(case.sub_questions)}
        emit(
            {
                "trace_id": trace_id,
                "event": "node_end",
                "node": "planner",
                "payload": {
                    "status": "completed",
                    "fixed": True,
                    "sub_question_count": len(case.sub_questions),
                    "latency_ms": (time.perf_counter() - started) * 1000,
                },
            }
        )
        return result

    return fixed_planner


def _disabled_critic_node(state: ResearchState) -> dict[str, object]:
    trace_id = state["trace_id"]
    emit(
        {
            "trace_id": trace_id,
            "event": "node_start",
            "node": "critic",
            "payload": {"enabled": False},
        }
    )
    emit(
        {
            "trace_id": trace_id,
            "event": "node_end",
            "node": "critic",
            "payload": {
                "status": "skipped",
                "enabled": False,
                "latency_ms": 0.0,
            },
        }
    )
    return {
        "quality_score": 1.0,
        "quality_history": [],
        "critique": "Critic disabled by the evaluation group.",
        "missing_aspects": [],
    }


def _cached_web_searcher(
    *,
    cache: QueryCache,
    settings: Settings,
    trace_id: str,
):
    def cached_search(query: str, max_results: int = 3) -> list[SearchResult]:
        resolved = cache.resolve(
            query,
            provider=settings.search_provider,
            max_results=max_results,
            fetcher=web_search if cache.mode == "record" else None,
        )
        emit(
            {
                "trace_id": trace_id,
                "event": "web_cache",
                "node": "researcher",
                "payload": {
                    "query": query,
                    "snapshot_id": cache.snapshot_id,
                    "cache_key": resolved.cache_key,
                    "from_cache": resolved.from_cache,
                    "mode": cache.mode,
                },
            }
        )
        return resolved.results

    return cached_search


def _task_record(
    task: PlannedTask,
    *,
    execution: TaskExecution,
    snapshot_id: str,
    cache_mode: CacheMode,
    settings: Settings,
) -> dict[str, Any]:
    state = execution.final_state
    final_report = state.get("final_report", "")
    status = "completed" if final_report.strip() else "failed"
    trace_summary = dict(execution.trace_summary)
    researcher_data = trace_summary.get("by_node", {}).get("researcher", {})
    researcher_latency_ms = (
        researcher_data.get("latency_ms", 0.0)
        if isinstance(researcher_data, Mapping)
        else 0.0
    )
    return {
        "schema_version": 1,
        "case_id": task.case.case_id,
        "case_type": task.case.case_type,
        "topic": task.case.topic,
        "sub_questions": list(task.case.sub_questions),
        "must_cover_ids": list(task.case.must_cover_ids),
        "must_cover": [list(concept) for concept in task.case.must_cover],
        "group": task.group.name,
        "track": task.group.track,
        "round_index": task.round_index,
        "concurrency": task.group.concurrency,
        "critic_enabled": task.group.critic_enabled,
        "planner_mode": "fixed_dataset",
        "cache_snapshot_id": snapshot_id,
        "cache_mode": cache_mode,
        "search_provider": settings.search_provider,
        "model_name": settings.model_name,
        "quality_threshold": settings.quality_threshold,
        "max_revision": settings.max_revision,
        "status": status,
        "final_report": final_report,
        "valid_citation_ids": list(_valid_citation_ids(state.get("citations", {}))),
        "quality_score": state.get("quality_score", 0.0),
        "quality_history": list(state.get("quality_history", [])),
        "fallback_queries": list(state.get("fallback_queries", [])),
        "errors": list(state.get("errors", [])),
        "researcher_latency_ms": researcher_latency_ms,
        "trace_id": state.get("trace_id", ""),
        "trace_summary": trace_summary,
        "web_cache_events": [dict(event) for event in execution.cache_events],
    }


def _valid_citation_ids(
    citations: Mapping[str, Sequence[Citation]],
) -> tuple[str, ...]:
    seen: set[tuple[str, str]] = set()
    count = 0
    for items in citations.values():
        for item in items:
            key = (item["source"], item["origin"])
            if key in seen:
                continue
            seen.add(key)
            count += 1
    return tuple(str(index) for index in range(1, count + 1))


def _plan_tasks(
    cases: Sequence[OrchestrationCase],
    *,
    groups: Sequence[ExperimentGroup],
    rounds: int,
) -> tuple[PlannedTask, ...]:
    tasks: list[PlannedTask] = []
    for round_index in range(1, rounds + 1):
        for case in cases:
            for group in groups:
                if group.track == "P" and case.case_type != "kb":
                    continue
                tasks.append(
                    PlannedTask(
                        case=case,
                        group=group,
                        round_index=round_index,
                    )
                )
    return tuple(tasks)


def _group_specs(
    groups: Sequence[OrchestrationGroup],
    settings: Settings,
) -> tuple[ExperimentGroup, ...]:
    specs = {
        "P1": ExperimentGroup("P1", "P", 1, False),
        "P2": ExperimentGroup("P2", "P", settings.research_concurrency, False),
        "Q1": ExperimentGroup("Q1", "Q", settings.research_concurrency, False),
        "Q2": ExperimentGroup("Q2", "Q", settings.research_concurrency, True),
    }
    return tuple(specs[group] for group in groups)


def _select_cases(
    cases: Sequence[OrchestrationCase],
    *,
    case_ids: Sequence[str] | None,
    case_limit: int | None,
) -> tuple[OrchestrationCase, ...]:
    selected = list(cases)
    if case_ids is not None:
        if isinstance(case_ids, str) or not case_ids:
            raise ValueError("case_ids must contain at least one case id")
        requested = {case_id.strip() for case_id in case_ids if case_id.strip()}
        if len(requested) != len(case_ids):
            raise ValueError("case_ids must contain unique non-empty values")
        known = {case.case_id for case in cases}
        unknown = requested.difference(known)
        if unknown:
            raise ValueError(f"unknown orchestration cases: {', '.join(sorted(unknown))}")
        selected = [case for case in cases if case.case_id in requested]
    if case_limit is not None:
        _require_positive_int("case_limit", case_limit)
        selected = selected[:case_limit]
    if not selected:
        raise ValueError("no orchestration cases selected")
    return tuple(selected)


def _normalize_groups(groups: Sequence[str]) -> tuple[OrchestrationGroup, ...]:
    if isinstance(groups, str) or not groups:
        raise ValueError("groups must contain at least one orchestration group")
    requested = {str(group).upper() for group in groups}
    unknown = requested.difference(DEFAULT_GROUPS)
    if unknown:
        raise ValueError(f"unknown orchestration groups: {', '.join(sorted(unknown))}")
    return tuple(group for group in DEFAULT_GROUPS if group in requested)


def _load_existing_records(
    path: Path,
    *,
    resume: bool,
) -> tuple[dict[str, Any], ...]:
    if not path.exists():
        return ()
    if not path.is_file():
        raise ValueError(f"raw output is not a file: {path}")
    if not resume:
        raise FileExistsError(f"orchestration raw output already exists: {path}")
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise ValueError(f"blank raw JSONL line: {path}:{line_number}")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid raw JSONL: {path}:{line_number}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"raw JSONL record must be an object: {path}:{line_number}")
        TaskObservation.from_raw(raw)
        key = _record_key(raw)
        if key in seen:
            raise ValueError(f"duplicate task record in raw output: {key}")
        seen.add(key)
        records.append(raw)
    return tuple(records)


def _record_key(record: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(record.get("case_id", "")),
        str(record.get("group", "")),
        int(record.get("round_index", 0)),
    )


def _validate_resume_identity(
    record: Mapping[str, Any],
    *,
    task: PlannedTask,
    snapshot_id: str,
    cache_mode: CacheMode,
    settings: Settings,
) -> None:
    expected = {
        "case_type": task.case.case_type,
        "topic": task.case.topic,
        "sub_questions": list(task.case.sub_questions),
        "must_cover_ids": list(task.case.must_cover_ids),
        "must_cover": [list(concept) for concept in task.case.must_cover],
        "track": task.group.track,
        "concurrency": task.group.concurrency,
        "critic_enabled": task.group.critic_enabled,
        "planner_mode": "fixed_dataset",
        "cache_snapshot_id": snapshot_id,
        "cache_mode": cache_mode,
        "search_provider": settings.search_provider,
        "model_name": settings.model_name,
        "quality_threshold": settings.quality_threshold,
        "max_revision": settings.max_revision,
    }
    for name, value in expected.items():
        if record.get(name) != value:
            raise ValueError(
                f"resume record identity mismatch for {task.key}: field={name}"
            )


def _append_jsonl_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(serialized + "\n")
        output.flush()
        os.fsync(output.fileno())


def _parse_case(
    raw: Mapping[str, Any],
    *,
    line_number: int,
) -> OrchestrationCase:
    case_id = _required_text(raw, "id", line_number=line_number)
    raw_type = _required_text(raw, "type", line_number=line_number).lower()
    if raw_type not in {"kb", "web", "mixed"}:
        raise OrchestrationDatasetError(f"invalid case type at line {line_number}")
    topic = _required_text(raw, "topic", line_number=line_number)
    sub_questions = _required_text_sequence(
        raw.get("sub_questions"),
        name="sub_questions",
        line_number=line_number,
    )
    raw_concepts = raw.get("must_cover")
    if not isinstance(raw_concepts, list) or not raw_concepts:
        raise OrchestrationDatasetError(
            f"must_cover must be a non-empty list at line {line_number}"
        )
    concept_ids: list[str] = []
    concepts: list[CoverConcept] = []
    for item in raw_concepts:
        if not isinstance(item, Mapping):
            raise OrchestrationDatasetError(
                f"must_cover item must be an object at line {line_number}"
            )
        concept_id = _required_text(item, "id", line_number=line_number)
        if concept_id in concept_ids:
            raise OrchestrationDatasetError(
                f"duplicate must_cover id at line {line_number}: {concept_id}"
            )
        concept_ids.append(concept_id)
        concepts.append(
            _required_text_sequence(
                item.get("any_of"),
                name="must_cover.any_of",
                line_number=line_number,
            )
        )
    return OrchestrationCase(
        case_id=case_id,
        case_type=cast(CaseType, raw_type),
        topic=topic,
        sub_questions=sub_questions,
        must_cover_ids=tuple(concept_ids),
        must_cover=tuple(concepts),
    )


def _required_text(
    raw: Mapping[str, Any],
    name: str,
    *,
    line_number: int,
) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise OrchestrationDatasetError(
            f"{name} must be a non-empty string at line {line_number}"
        )
    return value.strip()


def _required_text_sequence(
    value: Any,
    *,
    name: str,
    line_number: int,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise OrchestrationDatasetError(
            f"{name} must be a non-empty list at line {line_number}"
        )
    normalized = tuple(
        item.strip() for item in value if isinstance(item, str) and item.strip()
    )
    if len(normalized) != len(value) or len(set(normalized)) != len(normalized):
        raise OrchestrationDatasetError(
            f"{name} must contain unique non-empty strings at line {line_number}"
        )
    return normalized


def _validate_live_execution(
    *,
    live: bool,
    cache_mode: CacheMode,
    settings: Settings,
) -> None:
    if not live:
        raise ValueError("real orchestration execution requires explicit live=True")
    if not settings.trace_enabled:
        raise ValueError("real orchestration execution requires TRACE_ENABLED=true")
    settings.require_deepseek_api_key()
    if cache_mode == "record" and settings.search_provider == "tavily":
        settings.require_tavily_api_key()


def _require_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _no_op_stream_writer(_: Any) -> None:
    return None


__all__ = [
    "CacheWarmupReport",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_GROUPS",
    "DEFAULT_RAW_DIR",
    "DEFAULT_WEB_CACHE_DIR",
    "ExperimentGroup",
    "OrchestrationCase",
    "OrchestrationDatasetError",
    "OrchestrationGroup",
    "OrchestrationRunReport",
    "PlannedTask",
    "TaskExecution",
    "TaskExecutor",
    "default_raw_output_path",
    "load_orchestration_cases",
    "prewarm_initial_query_cache",
    "run_orchestration_evaluation",
]
