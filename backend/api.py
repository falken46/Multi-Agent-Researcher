"""FastAPI 后端入口。"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from backend.streaming import stream_research_progress

STREAMLIT_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]


class ResearchRequest(BaseModel):
    """研究任务请求体。"""

    topic: str = Field(..., min_length=1)

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("topic must not be empty")
        return normalized_value


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
            stream_research_progress(request.topic),
            media_type="text/event-stream",
        )

    return app


app = create_app()

__all__ = ["ResearchRequest", "app", "create_app"]
