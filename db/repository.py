"""数据访问层（Phase 19）。

**设计约束：模型永远不生成 SQL。**

LLM 只输出结构化字段（``ResearchState`` 里的那些），SQL 语句由本模块用固定的
ORM 调用装配。模型碰不到 SQL 文本，也就没有注入面。

> 这与实习里「白名单 + 字段字典 + 模板装配只读 SQL」是同一套思维，
> 但**实现是新写的、更简单的版本**（这里靠 ORM，那边是自建装配器）。
> 对外表述不得把两者混为一谈 —— 见 `UPGRADE_V3.md` 与秋招总索引 §6 红线。

所有写入都通过 :func:`save_research_run` 一个入口，在**同一个事务**里完成，
中途异常整体回滚，不留半条记录。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Citation, Report, ResearchTask, SubQuestion

logger = logging.getLogger(__name__)

_MAX_SNIPPET = 500


def get_task_by_thread_id(session: Session, thread_id: str) -> ResearchTask | None:
    return session.scalar(
        select(ResearchTask).where(ResearchTask.thread_id == thread_id)
    )


def save_research_run(
    session: Session,
    *,
    thread_id: str,
    state: Mapping[str, Any],
    status: str = "completed",
    actor: str | None = None,
    model_name: str | None = None,
    vector_backend: str | None = None,
    finished_at: datetime | None = None,
) -> ResearchTask:
    """把一次研究任务的最终状态落库，按 ``thread_id`` 幂等。

    幂等的必要性：LangGraph 支持断点续跑，同一个 ``thread_id`` 可能被恢复执行多次。
    如果每次都 insert，一个任务会在库里变成好几条，统计就废了。所以这里先查后写 ——
    已存在就更新那条，子问题与报告按需追加。

    返回落库后的 :class:`ResearchTask`。
    """
    task = get_task_by_thread_id(session, thread_id)
    usage = dict(state.get("usage") or {})

    if task is None:
        task = ResearchTask(thread_id=thread_id, topic=str(state.get("topic", "")))
        session.add(task)

    task.topic = str(state.get("topic", "")) or task.topic
    task.trace_id = state.get("trace_id") or task.trace_id
    task.status = status
    task.actor = actor or task.actor
    task.model_name = model_name or task.model_name
    task.vector_backend = vector_backend or task.vector_backend
    # 任务开始时 state 里还没有 usage。此时新记录沿用字段默认值，恢复已有任务时
    # 则保留上一段执行已经记录的用量，避免一次 resume 把历史统计先清成 0。
    if usage:
        task.prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        task.completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        task.total_tokens = int(usage.get("total_tokens", 0) or 0)
        task.total_cost = float(usage.get("total_cost", 0.0) or 0.0)
        task.llm_calls = int(usage.get("llm_calls", 0) or 0)
        task.total_latency_ms = float(usage.get("total_latency_ms", 0.0) or 0.0)
    if status == "running":
        # 失败任务被恢复时重新进入运行态，旧 finished_at 不能继续保留。
        task.finished_at = None
    elif status in {"completed", "failed"}:
        task.finished_at = finished_at or datetime.now(timezone.utc)

    # flush 而不是 commit：把 task.id 生成出来供下面的外键使用，
    # 但事务边界仍由 session_scope 统一控制，异常时整体回滚。
    session.flush()

    sub_question_ids = _sync_sub_questions(session, task, state)
    _append_report(session, task, state, sub_question_ids)
    session.flush()
    return task


def _sync_sub_questions(
    session: Session,
    task: ResearchTask,
    state: Mapping[str, Any],
) -> dict[str, int]:
    """写入子问题，返回 ``子问题文本 -> id`` 映射，供 citation 关联。"""
    questions: Sequence[str] = list(state.get("sub_questions") or [])
    fallback = set(state.get("fallback_queries") or [])

    existing = {row.seq: row for row in task.sub_questions}
    mapping: dict[str, int] = {}

    for seq, question in enumerate(questions):
        row = existing.get(seq)
        if row is None:
            row = SubQuestion(task_id=task.id, seq=seq, question=question)
            session.add(row)
        else:
            row.question = question
        row.fallback_triggered = question in fallback
        session.flush()
        mapping[question] = row.id

    return mapping


def _append_report(
    session: Session,
    task: ResearchTask,
    state: Mapping[str, Any],
    sub_question_ids: Mapping[str, int],
) -> Report | None:
    """追加一版报告及其引用。空报告不落库。"""
    content = str(state.get("final_report") or "").strip()
    if not content:
        return None

    revision_count = int(state.get("revision_count", 0) or 0)
    # 同一个 revision 已经存过就不重复追加（恢复续跑会重复走到收尾节点）
    for row in task.reports:
        if row.revision_count == revision_count and row.content == content:
            return row

    report = Report(
        task_id=task.id,
        content=content,
        quality_score=_as_float(state.get("quality_score")),
        critique=state.get("critique") or None,
        revision_count=revision_count,
    )
    session.add(report)
    session.flush()

    _append_citations(session, report, state, sub_question_ids)
    return report


def _append_citations(
    session: Session,
    report: Report,
    state: Mapping[str, Any],
    sub_question_ids: Mapping[str, int],
) -> None:
    """展平 ``citations`` 并按检索名次落库。

    ``state["citations"]`` 的结构是 ``{子问题: [Citation, ...]}``，
    列表顺序就是当时的检索名次，所以 ``rank`` 直接取下标 —— 这是**当时那次检索的事实**，
    索引重建后就再也复原不了，必须和结论一起冻结。
    """
    grouped: Mapping[str, Sequence[Mapping[str, Any]]] = state.get("citations") or {}
    for question, items in grouped.items():
        for rank, item in enumerate(items):
            session.add(
                Citation(
                    report_id=report.id,
                    sub_question_id=sub_question_ids.get(question),
                    source=str(item.get("source", "")),
                    origin=str(item.get("origin", "kb")),
                    snippet=(item.get("snippet") or "")[:_MAX_SNIPPET] or None,
                    rank=rank,
                    # 上游 Citation 目前不携带分数，取到就存、取不到留空。
                    # 留空比填 0 诚实 —— 0 会被误读成"相关度为零"。
                    retrieval_score=_as_float(item.get("score")),
                )
            )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["get_task_by_thread_id", "save_research_run"]
