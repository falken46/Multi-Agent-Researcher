"""Multi-Agent 研究流程的共享状态定义。"""

from __future__ import annotations

from typing import Literal, TypedDict


class Citation(TypedDict):
    """Researcher 产出的可追溯证据。"""

    source: str
    origin: Literal["kb", "web"]
    snippet: str


class Usage(TypedDict):
    """从 trace 汇总得到的任务资源消耗。"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_cost: float
    llm_calls: int
    total_latency_ms: float


class ResearchState(TypedDict):
    """Planner、Researcher、Critic 与 Writer 共享的状态 schema。"""

    topic: str
    sub_questions: list[str]
    research_results: dict[str, str]
    final_report: str
    errors: list[str]
    retry_count: int
    citations: dict[str, list[Citation]]
    critique: str
    quality_score: float
    quality_history: list[float]
    missing_aspects: list[str]
    revision_count: int
    trace_id: str
    usage: Usage
    fallback_queries: list[str]


__all__ = ["Citation", "ResearchState", "Usage"]
