"""LangGraph 状态机定义。"""

from __future__ import annotations

import logging
from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from agents.planner import planner_node
from agents.researcher import researcher_node
from agents.state import ResearchState
from agents.writer import writer_node
from core.config import get_settings
from core.trace import new_trace_id

logger = logging.getLogger(__name__)

GraphNode = Callable[[ResearchState], dict[str, object]]


def create_initial_state(topic: str) -> ResearchState:
    """根据用户主题创建 LangGraph 初始状态。"""
    return {
        "topic": topic,
        "sub_questions": [],
        "research_results": {},
        "final_report": "",
        "errors": [],
        "retry_count": 0,
        "trace_id": new_trace_id(),
    }


def should_retry(state: ResearchState) -> str:
    """根据 Researcher 输出决定重试或继续。"""
    if not state.get("sub_questions"):
        logger.info("graph route continue: no sub_questions")
        return "continue"
    if state.get("research_results"):
        logger.info("graph route continue: research_results present")
        return "continue"

    retry_count = state.get("retry_count", 0)
    max_retry = get_settings().max_retry
    if retry_count < max_retry:
        logger.info(
            "graph route retry: retry_count=%s max_retry=%s",
            retry_count,
            max_retry,
        )
        return "retry"

    logger.info(
        "graph route continue: retry_count=%s max_retry=%s",
        retry_count,
        max_retry,
    )
    return "continue"


def build_graph(
    planner: GraphNode = planner_node,
    researcher: GraphNode = researcher_node,
    writer: GraphNode = writer_node,
):
    """构建 Planner -> Researcher -> Writer 的 LangGraph 状态机。"""
    graph_builder = StateGraph(ResearchState)
    graph_builder.add_node("planner", planner)
    graph_builder.add_node("researcher", researcher)
    graph_builder.add_node("writer", writer)

    graph_builder.add_edge(START, "planner")
    graph_builder.add_edge("planner", "researcher")
    graph_builder.add_conditional_edges(
        "researcher",
        should_retry,
        {"retry": "researcher", "continue": "writer"},
    )
    graph_builder.add_edge("writer", END)
    return graph_builder.compile()

graph = build_graph()

__all__ = ["build_graph", "create_initial_state", "graph", "should_retry"]
