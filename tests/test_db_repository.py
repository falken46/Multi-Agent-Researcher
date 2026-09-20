"""数据库层测试（Phase 19）。

跑在 SQLite 上：模型刻意保持方言中立，所以同一套测试在 CI 的 PostgreSQL
service container 里也能跑（见 `.github/workflows/ci.yml`）。
本地不需要起数据库，保住"无外部依赖也能跑完整离线测试"。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from core.config import Settings
from db import repository
from db.models import Citation, Report, ResearchTask, SubQuestion
from db.persistence import persist_research_run
from db.session import (
    create_all,
    drop_all,
    is_enabled,
    reset_engine_cache,
    session_scope,
)


@pytest.fixture
def db_settings(runtime_dir: Path) -> Settings:
    """测试用数据库。

    默认是一次性 SQLite 文件，本地与常规 CI 不需要起任何服务。

    设了环境变量 ``TEST_DATABASE_URL`` 时改用它 —— CI 里用这条把**同一套测试**
    再对着真实 PostgreSQL 跑一遍。模型写成方言中立只是意图，
    「在 SQLite 上过」不等于「在 PostgreSQL 上过」，得真跑才算数。
    """
    external_url = os.environ.get("TEST_DATABASE_URL", "").strip()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        _env_file=None,
        database_url=external_url
        or f"sqlite:///{(runtime_dir / 'test.db').as_posix()}",
    )
    reset_engine_cache()
    if external_url:
        # 外部库是复用的，每个用例前后都要清干净，避免用例互相污染
        drop_all(settings)
    create_all(settings)
    yield settings
    if external_url:
        drop_all(settings)
    reset_engine_cache()


def _state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "topic": "RRF 融合为什么有效",
        "trace_id": "trace-001",
        "sub_questions": ["RRF 的公式是什么", "为什么不用加权求和"],
        "final_report": "报告正文",
        "critique": "覆盖充分",
        "quality_score": 0.82,
        "revision_count": 0,
        "fallback_queries": ["为什么不用加权求和"],
        "citations": {
            "RRF 的公式是什么": [
                {"source": "13_rrf.md", "origin": "kb", "snippet": "RRF 定义", "score": 0.81},
                {"source": "36_rank_fusion.md", "origin": "kb", "snippet": "融合方法"},
            ],
            "为什么不用加权求和": [
                {"source": "https://example.com/a", "origin": "web", "snippet": "网页"},
            ],
        },
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 45,
            "total_tokens": 165,
            "total_cost": 0.0031,
            "llm_calls": 4,
            "total_latency_ms": 1820.5,
        },
    }
    base.update(overrides)
    return base


# ---------- 基础落库 ----------


def test_save_research_run_persists_full_chain(db_settings: Settings) -> None:
    with session_scope(db_settings) as session:
        repository.save_research_run(
            session, thread_id="t-1", state=_state(), model_name="deepseek-v4-flash"
        )

    with session_scope(db_settings) as session:
        task = repository.get_task_by_thread_id(session, "t-1")
        assert task is not None
        assert task.topic == "RRF 融合为什么有效"
        assert task.trace_id == "trace-001"
        assert task.status == "completed"
        assert task.model_name == "deepseek-v4-flash"
        assert task.total_tokens == 165
        assert task.llm_calls == 4
        assert task.finished_at is not None

        assert [q.question for q in task.sub_questions] == [
            "RRF 的公式是什么",
            "为什么不用加权求和",
        ]
        assert [q.fallback_triggered for q in task.sub_questions] == [False, True]

        assert len(task.reports) == 1
        report = task.reports[0]
        assert report.content == "报告正文"
        assert report.quality_score == pytest.approx(0.82)

        citations = session.scalars(
            select(Citation).where(Citation.report_id == report.id)
        ).all()
        assert len(citations) == 3
        assert {c.origin for c in citations} == {"kb", "web"}


def test_citation_stores_rank_and_score(db_settings: Settings) -> None:
    """rank 与 retrieval_score 是本表的设计核心，必须真的落下去。

    只存"引用了这个来源"，事后就回答不了「当时为什么是这条排第一」。
    """
    with session_scope(db_settings) as session:
        repository.save_research_run(session, thread_id="t-2", state=_state())

    with session_scope(db_settings) as session:
        rows = session.scalars(
            select(Citation).where(Citation.source == "13_rrf.md")
        ).all()
        assert len(rows) == 1
        assert rows[0].rank == 0
        assert rows[0].retrieval_score == pytest.approx(0.81)

        # 上游没给分数时留空，而不是填 0 —— 0 会被误读成"相关度为零"
        no_score = session.scalar(
            select(Citation).where(Citation.source == "36_rank_fusion.md")
        )
        assert no_score is not None
        assert no_score.rank == 1
        assert no_score.retrieval_score is None


def test_citation_links_back_to_sub_question(db_settings: Settings) -> None:
    with session_scope(db_settings) as session:
        repository.save_research_run(session, thread_id="t-3", state=_state())

    with session_scope(db_settings) as session:
        citation = session.scalar(
            select(Citation).where(Citation.source == "13_rrf.md")
        )
        sub_question = session.get(SubQuestion, citation.sub_question_id)
        assert sub_question is not None
        assert sub_question.question == "RRF 的公式是什么"


# ---------- 幂等 ----------


def test_same_thread_id_is_idempotent(db_settings: Settings) -> None:
    """断点续跑会让同一个 thread_id 多次走到收尾，不能落成多条任务。"""
    for _ in range(3):
        with session_scope(db_settings) as session:
            repository.save_research_run(session, thread_id="t-4", state=_state())

    with session_scope(db_settings) as session:
        tasks = session.scalars(
            select(ResearchTask).where(ResearchTask.thread_id == "t-4")
        ).all()
        assert len(tasks) == 1
        assert len(tasks[0].sub_questions) == 2
        # 同一版报告内容不重复追加
        assert len(tasks[0].reports) == 1


def test_revision_creates_a_new_report_version(db_settings: Settings) -> None:
    with session_scope(db_settings) as session:
        repository.save_research_run(session, thread_id="t-5", state=_state())
    with session_scope(db_settings) as session:
        repository.save_research_run(
            session,
            thread_id="t-5",
            state=_state(final_report="返工后的报告", revision_count=1),
        )

    with session_scope(db_settings) as session:
        task = repository.get_task_by_thread_id(session, "t-5")
        assert len(task.reports) == 2
        assert [r.revision_count for r in task.reports] == [0, 1]


def test_empty_report_is_not_persisted(db_settings: Settings) -> None:
    with session_scope(db_settings) as session:
        repository.save_research_run(
            session, thread_id="t-6", state=_state(final_report="   ")
        )

    with session_scope(db_settings) as session:
        task = repository.get_task_by_thread_id(session, "t-6")
        assert task is not None
        assert task.reports == []


# ---------- 事务与级联 ----------


def test_transaction_rolls_back_on_error(db_settings: Settings) -> None:
    """中途异常不留半条记录。"""
    with pytest.raises(RuntimeError):
        with session_scope(db_settings) as session:
            repository.save_research_run(session, thread_id="t-7", state=_state())
            raise RuntimeError("boom")

    with session_scope(db_settings) as session:
        assert repository.get_task_by_thread_id(session, "t-7") is None
        assert session.scalars(select(Report)).all() == []
        assert session.scalars(select(Citation)).all() == []


def test_deleting_task_cascades(db_settings: Settings) -> None:
    with session_scope(db_settings) as session:
        repository.save_research_run(session, thread_id="t-8", state=_state())

    with session_scope(db_settings) as session:
        task = repository.get_task_by_thread_id(session, "t-8")
        session.delete(task)

    with session_scope(db_settings) as session:
        assert session.scalars(select(ResearchTask)).all() == []
        assert session.scalars(select(SubQuestion)).all() == []
        assert session.scalars(select(Report)).all() == []
        assert session.scalars(select(Citation)).all() == []


# ---------- 未配置数据库时的降级 ----------


def test_persistence_is_skipped_when_database_url_is_empty() -> None:
    """没配 DATABASE_URL 时静默跳过 —— 保住"无外部依赖也能跑"。"""
    settings = Settings(_env_file=None, database_url="")
    assert is_enabled(settings) is False
    assert persist_research_run(thread_id="t-9", state=_state(), settings=settings) is None


def test_persist_swallows_errors_and_returns_none(runtime_dir: Path) -> None:
    """落库失败必须 fail-open：只告警，不打断主流程。

    这里指向一个不存在的目录制造失败。对照：合规审查版同一层是 fail-closed。
    """
    reset_engine_cache()
    settings = Settings(
        _env_file=None,
        database_url="sqlite:////nonexistent-dir-xyz/does-not-exist.db",
    )
    assert persist_research_run(thread_id="t-10", state=_state(), settings=settings) is None
    reset_engine_cache()


def test_persist_writes_when_enabled(db_settings: Settings) -> None:
    task_id = persist_research_run(
        thread_id="t-11", state=_state(), settings=db_settings
    )
    assert task_id is not None

    with session_scope(db_settings) as session:
        task = repository.get_task_by_thread_id(session, "t-11")
        assert task is not None
        assert task.id == task_id
