"""FastAPI 后端入口。"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from sse_starlette.sse import EventSourceResponse

from backend.streaming import stream_research_progress

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
    async def research(request: ResearchRequest) -> EventSourceResponse:
        return EventSourceResponse(
            stream_research_progress(
                request.topic,
                thread_id=request.thread_id,
                resume=request.resume,
            ),
            media_type="text/event-stream",
        )

    return app


app = create_app()

__all__ = ["ResearchRequest", "app", "create_app"]
