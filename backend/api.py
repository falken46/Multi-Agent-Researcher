"""FastAPI 后端入口。"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from backend.auth import require_actor
from backend.streaming import stream_research_progress
from db import repository
from db.models import Citation, ResearchTask
from db.session import is_enabled, session_scope

STREAMLIT_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]


class ResearchRequest(BaseModel):
    """研究任务请求体。"""

    topic: str = Field(..., min_length=1)
    thread_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$",
    )
    resume: bool = False

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("topic must not be empty")
        return normalized_value

    @field_validator("thread_id")
    @classmethod
    def validate_thread_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip()

    @model_validator(mode="after")
    def validate_resume(self) -> ResearchRequest:
        if self.resume and not self.thread_id:
            raise ValueError("thread_id is required when resume is true")
        return self


Actor = Annotated[str, Depends(require_actor)]


def _require_database() -> None:
    """查询类接口需要落库才有意义。

    未配置 ``DATABASE_URL`` 时返回 503 而不是空列表 —— 空列表会被误读成
    "确实没有任务"，而真相是"这个部署根本没开落库"。
    """
    if not is_enabled():
        raise HTTPException(
            status_code=503,
            detail="task history requires DATABASE_URL to be configured",
        )


def _task_to_dict(task: ResearchTask) -> dict[str, Any]:
    return {
        "id": task.id,
        "topic": task.topic,
        "thread_id": task.thread_id,
        "trace_id": task.trace_id,
        "status": task.status,
        "model_name": task.model_name,
        "vector_backend": task.vector_backend,
        "usage": {
            "prompt_tokens": task.prompt_tokens,
            "completion_tokens": task.completion_tokens,
            "total_tokens": task.total_tokens,
            "total_cost": task.total_cost,
            "llm_calls": task.llm_calls,
            "total_latency_ms": task.total_latency_ms,
        },
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "finished_at": task.finished_at.isoformat() if task.finished_at else None,
    }


def create_app() -> FastAPI:
    """创建 FastAPI 应用。"""
    app = FastAPI(title="Multi-Agent Research Assistant")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=STREAMLIT_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/research")
    async def research(request: ResearchRequest, actor: Actor) -> EventSourceResponse:
        return EventSourceResponse(
            stream_research_progress(
                request.topic,
                thread_id=request.thread_id,
                resume=request.resume,
                actor=actor,
            ),
            media_type="text/event-stream",
        )

    @app.get("/tasks")
    async def list_tasks(
        actor: Actor,
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict[str, Any]:
        """按开始时间倒序列出历史任务。"""
        _require_database()
        with session_scope() as session:
            rows = session.scalars(
                select(ResearchTask)
                .order_by(ResearchTask.started_at.desc())
                .limit(limit)
            ).all()
            return {"actor": actor, "count": len(rows), "tasks": [_task_to_dict(r) for r in rows]}

    @app.get("/tasks/{thread_id}")
    async def get_task(thread_id: str, actor: Actor) -> dict[str, Any]:
        """查询单个任务及其子问题。"""
        _require_database()
        with session_scope() as session:
            task = repository.get_task_by_thread_id(session, thread_id)
            if task is None:
                raise HTTPException(status_code=404, detail="task not found")
            payload = _task_to_dict(task)
            payload["sub_questions"] = [
                {
                    "seq": item.seq,
                    "question": item.question,
                    "fallback_triggered": item.fallback_triggered,
                }
                for item in task.sub_questions
            ]
            payload["report_count"] = len(task.reports)
            return payload

    @app.get("/tasks/{thread_id}/report")
    async def get_report(thread_id: str, actor: Actor) -> dict[str, Any]:
        """取最新一版报告及其引用来源（含检索名次）。"""
        _require_database()
        with session_scope() as session:
            task = repository.get_task_by_thread_id(session, thread_id)
            if task is None:
                raise HTTPException(status_code=404, detail="task not found")
            if not task.reports:
                raise HTTPException(status_code=404, detail="task has no report yet")
            report = task.reports[-1]
            citations = session.scalars(
                select(Citation)
                .where(Citation.report_id == report.id)
                .order_by(Citation.rank)
            ).all()
            return {
                "thread_id": thread_id,
                "revision_count": report.revision_count,
                "quality_score": report.quality_score,
                "critique": report.critique,
                "content": report.content,
                "citations": [
                    {
                        "rank": item.rank,
                        "source": item.source,
                        "origin": item.origin,
                        # 名次一定有；分数上游未必带下来，没有就是 null，不填 0
                        "retrieval_score": item.retrieval_score,
                        "snippet": item.snippet,
                    }
                    for item in citations
                ],
            }

    return app


app = create_app()

__all__ = ["ResearchRequest", "app", "create_app"]
