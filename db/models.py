"""业务数据的 ORM 模型（Phase 19）。

v2 之前全项目零业务数据持久化：一次 ``/research`` 跑完，问题、子问题、报告和引用
全都只活在内存与 trace 文件里，任务结束就没了。这里补上四张表。

**方言中立**：只用 SQLAlchemy 通用类型（``String`` / ``Text`` / ``JSON`` / ``Numeric`` …），
不用 PostgreSQL 专有类型（如 ``JSONB`` / ``ARRAY``）。这样同一套模型可以：

- 生产与 compose 里跑 PostgreSQL
- 本地与 CI 里跑 SQLite，测试不需要起数据库

代价是放弃了 JSONB 的索引能力；本项目的 JSON 字段只做读取展示，不做条件查询，够用。
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    """带时区的当前时间。

    不用 ``datetime.utcnow()`` —— 那个返回的是 naive datetime（不带时区信息），
    存进库再读出来就分不清是本地时间还是 UTC，跨时区必然出错。
    """
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """所有表的公共基类。Alembic 通过 ``Base.metadata`` 发现表结构。"""


class ResearchTask(Base):
    """一次研究任务。与 LangGraph 的一个 ``thread_id`` 一一对应。"""

    __tablename__ = "research_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    topic: Mapped[str] = mapped_column(Text, nullable=False)
    # thread_id 是 LangGraph checkpoint 的键，也是"这次任务"的业务身份。
    # 唯一约束保证同一个 thread 不会落两条任务记录（恢复续跑时会重复写）。
    thread_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    trace_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    # 调用主体（Phase 20）。可观测缺了"谁在调"这一维，事后就回答不了
    # 「这次任务是谁发起的」。未开鉴权时为 anonymous。
    actor: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    vector_backend: Mapped[str | None] = mapped_column(String(32), nullable=True)

    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    llm_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    sub_questions: Mapped[list[SubQuestion]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="SubQuestion.seq",
    )
    reports: Mapped[list[Report]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="Report.id",
    )


class SubQuestion(Base):
    """Planner 拆出的一个子问题，以及它在检索时有没有触发过降级。"""

    __tablename__ = "sub_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(
        ForeignKey("research_tasks.id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    # 本地库召回不足时 Researcher 会降级联网，这个标记让事后能分析
    # "哪些子问题本地知识库答不了"。
    fallback_triggered: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    task: Mapped[ResearchTask] = relationship(back_populates="sub_questions")

    __table_args__ = (
        # 同一个任务里子问题序号不重复；重复写入会被数据库挡下，而不是悄悄多一行
        UniqueConstraint("task_id", "seq", name="uq_sub_questions_task_seq"),
    )


class Report(Base):
    """一次任务产出的报告。返工会产生多版，用 ``revision_count`` 区分。"""

    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(
        ForeignKey("research_tasks.id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    critique: Mapped[str | None] = mapped_column(Text, nullable=True)
    revision_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    task: Mapped[ResearchTask] = relationship(back_populates="reports")
    citations: Mapped[list[Citation]] = relationship(
        back_populates="report",
        cascade="all, delete-orphan",
        order_by="Citation.rank",
    )


class Citation(Base):
    """报告引用到的一条来源。

    **设计约束：必须存 ``rank``（和可得时的 ``retrieval_score``）。**

    只存"报告引用了这个来源"是不够的 —— 事后无法回答「当时为什么是这条排第一」。
    索引重建、模型升级之后，这个问题就永远答不了了。名次是当时那次检索的事实，
    必须和结论一起冻结下来。
    """

    __tablename__ = "citations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_id: Mapped[int] = mapped_column(
        ForeignKey("reports.id", ondelete="CASCADE"), nullable=False
    )
    # 属于哪个子问题的检索结果（可为空：报告级引用不挂在具体子问题上）
    sub_question_id: Mapped[int | None] = mapped_column(
        ForeignKey("sub_questions.id", ondelete="SET NULL"), nullable=True
    )
    source: Mapped[str] = mapped_column(Text, nullable=False)
    origin: Mapped[str] = mapped_column(String(16), nullable=False)  # kb | web
    snippet: Mapped[str | None] = mapped_column(Text, nullable=True)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    # 可为空：联网结果没有可比的检索分数，本地库结果才有
    retrieval_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    report: Mapped[Report] = relationship(back_populates="citations")

    __table_args__ = (
        Index("ix_citations_report", "report_id"),
        Index("ix_citations_origin", "origin"),
    )


__all__ = [
    "Base",
    "Citation",
    "Report",
    "ResearchTask",
    "SubQuestion",
]
