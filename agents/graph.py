"""带技术重试、质量反思回环与可注入 Checkpointer 的 LangGraph。"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.critic import critic_node
from agents.planner import planner_node
from agents.researcher import researcher_node
from agents.state import ResearchState
from agents.writer import writer_node
from core.config import get_settings
from core.trace import new_trace_id

logger = logging.getLogger(__name__)

NodeResult = dict[str, object]
GraphNode = Callable[[ResearchState], NodeResult | Awaitable[NodeResult]]


def create_initial_state(topic: str, *, trace_id: str | None = None) -> ResearchState:
    """根据用户主题创建完整且可持久化的 LangGraph 初始状态。"""
    return {
        "topic": topic,
        "sub_questions": [],
        "research_results": {},
        "final_report": "",
        "errors": [],
        "retry_count": 0,
        "citations": {},
        "critique": "",
        "quality_score": 0.0,
        "quality_history": [],
        "missing_aspects": [],
        "revision_count": 0,
        "trace_id": trace_id or new_trace_id(),
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "llm_calls": 0,
            "total_latency_ms": 0.0,
        },
        "fallback_queries": [],
    }


def should_continue(state: ResearchState) -> str:
    """Researcher 后按结果与技术重试次数路由。"""
    if state.get("research_results"):
        logger.info("graph route critic: research_results present")
        return "critic"

    retry_count = state.get("retry_count", 0)
    max_retry = get_settings().max_retry
    if state.get("sub_questions") and retry_count < max_retry:
        logger.info(
            "graph route retry: retry_count=%s max_retry=%s",
            retry_count,
            max_retry,
        )
        return "retry"

    logger.info(
        "graph route writer: no results retry_count=%s max_retry=%s",
        retry_count,
        max_retry,
    )
    return "writer"


def should_revise(state: ResearchState) -> str:
    """Critic 后执行质量阈值、硬上限、定向补查与停滞保护。"""
    settings = get_settings()
    score = state.get("quality_score", 0.0)
    revision_count = state.get("revision_count", 0)
    missing_aspects = state.get("missing_aspects", [])
    if score >= settings.quality_threshold:
        logger.info(
            "graph route writer: score=%.3f threshold=%.3f",
            score,
            settings.quality_threshold,
        )
        return "writer"
    if not missing_aspects:
        logger.info("graph route writer: no directed missing_aspects")
        return "writer"
    if revision_count >= settings.max_revision:
        logger.info(
            "graph route writer: revision_count=%s max_revision=%s",
            revision_count,
            settings.max_revision,
        )
        return "writer"
    if _quality_stalled(state):
        logger.info(
            "graph route writer: quality stalled history=%s",
            state.get("quality_history", []),
        )
        return "writer"
    logger.info(
        "graph route researcher: score=%.3f revision_count=%s missing=%s",
        score,
        revision_count,
        len(missing_aspects),
    )
    return "researcher"


def should_retry(state: ResearchState) -> str:
    """保留 v1 路由名的兼容入口。"""
    route = should_continue(state)
    return "continue" if route in {"critic", "writer"} else "retry"


def build_graph(
    planner: GraphNode = planner_node,
    researcher: GraphNode = researcher_node,
    critic: GraphNode = critic_node,
    writer: GraphNode = writer_node,
    *,
    checkpointer: Any | None = None,
    interrupt_after: list[str] | None = None,
):
    """构建 Planner → Researcher ⇄ Critic → Writer 状态机。"""
    graph_builder = StateGraph(ResearchState)
    graph_builder.add_node("planner", planner)
    graph_builder.add_node("researcher", researcher)
    graph_builder.add_node("critic", critic)
    graph_builder.add_node("writer", writer)

    graph_builder.add_edge(START, "planner")
    graph_builder.add_edge("planner", "researcher")
    graph_builder.add_conditional_edges(
        "researcher",
        should_continue,
        {"retry": "researcher", "critic": "critic", "writer": "writer"},
    )
    graph_builder.add_conditional_edges(
        "critic",
        should_revise,
        {"researcher": "researcher", "writer": "writer"},
    )
    graph_builder.add_edge("writer", END)
    return graph_builder.compile(
        checkpointer=checkpointer,
        interrupt_after=interrupt_after,
    )


def _quality_stalled(state: ResearchState) -> bool:
    revision_count = state.get("revision_count", 0)
    history = state.get("quality_history", [])
    if revision_count < 2 or len(history) < 3:
        return False
    return history[-1] <= max(history[:-1])


graph = build_graph()

__all__ = [
    "build_graph",
    "create_initial_state",
    "graph",
    "should_continue",
    "should_revise",
    "should_retry",
]
