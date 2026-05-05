"""Multi-Agent 研究流程的共享状态定义。"""

from __future__ import annotations

from typing import TypedDict


class ResearchState(TypedDict):
    """Planner, Researcher, Writer 共享的 LangGraph 状态 schema。"""

    topic: str
    sub_questions: list[str]
    research_results: dict[str, str]
    final_report: str
    errors: list[str]
    retry_count: int
