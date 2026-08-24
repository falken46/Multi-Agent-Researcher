"""Streamlit 前端入口,负责提交研究主题并渲染 SSE 进度。"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from typing import Any, TypedDict

import requests
import streamlit as st

from core.config import get_settings

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = (5.0, 600.0)
NODE_ORDER = ["planner", "researcher", "writer"]
NODE_LABELS = {
    "planner": "Planner",
    "researcher": "Researcher",
    "writer": "Writer",
}


class SSEEvent(TypedDict):
    """前端内部使用的 SSE 事件结构。"""

    event: str
    data: Any


class AgentStatus(TypedDict):
    """Agent 展示状态。"""

    status: str
    detail: str


class ViewState(TypedDict):
    """Streamlit 渲染所需的页面状态。"""

    topic: str
    agent_status: dict[str, AgentStatus]
    sub_questions: list[str]
    research_result_count: int
    final_report: str
    errors: list[str]
    retry_count: int
    events: list[dict[str, str]]


def create_view_state(topic: str = "") -> ViewState:
    """创建前端初始展示状态。"""
    return {
        "topic": topic,
        "agent_status": {
            node: {"status": "等待", "detail": ""}
            for node in NODE_ORDER
        },
        "sub_questions": [],
        "research_result_count": 0,
        "final_report": "",
        "errors": [],
        "retry_count": 0,
        "events": [],
    }


def parse_sse_lines(lines: Iterable[str | bytes]) -> Iterator[SSEEvent]:
    """解析 SSE 行流,返回结构化事件。"""
    event_name = "message"
    data_lines: list[str] = []

    for raw_line in lines:
        line = _normalize_sse_line(raw_line)
        if line == "":
            if data_lines:
                yield _build_sse_event(event_name, data_lines)
            event_name = "message"
            data_lines = []
            continue

        if line.startswith(":"):
            continue

        field_name, separator, field_value = line.partition(":")
        if not separator:
            continue
        if field_value.startswith(" "):
            field_value = field_value[1:]

        if field_name == "event":
            event_name = field_value or "message"
        elif field_name == "data":
            data_lines.append(field_value)

    if data_lines:
        yield _build_sse_event(event_name, data_lines)


def subscribe_research(
    topic: str,
    api_base_url: str = DEFAULT_API_BASE_URL,
    timeout: tuple[float, float] = REQUEST_TIMEOUT,
) -> Iterator[SSEEvent]:
    """调用后端研究接口,持续产出 SSE 事件。"""
    endpoint = f"{api_base_url.rstrip('/')}/research"
    with requests.post(
        endpoint,
        json={"topic": topic},
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        yield from parse_sse_lines(response.iter_lines(decode_unicode=True))


def apply_event_to_view_state(event: SSEEvent, view_state: ViewState) -> ViewState:
    """把后端事件合并到前端展示状态。"""
    event_name = event["event"]
    payload = event["data"] if isinstance(event["data"], dict) else {}
    node = str(payload.get("node", ""))

    _append_event_log(view_state, event_name=event_name, node=node, payload=payload)

    if event_name == "start":
        view_state["topic"] = str(payload.get("topic", view_state["topic"]))
        _mark_node(view_state, "planner", "运行中", "正在拆解研究问题")
        return view_state

    state_summary = payload.get("state")
    if isinstance(state_summary, dict):
        _apply_state_summary(view_state, state_summary)

    payload_status = str(payload.get("status", ""))
    if event_name == "progress" and node in NODE_LABELS:
        _mark_node(view_state, node, _display_status(payload_status), _node_detail(node, view_state))
        _apply_error_statuses(view_state)
        if payload_status not in {"failed", "error"}:
            _mark_next_node_running(view_state, node)
    elif event_name == "complete":
        if view_state["final_report"]:
            for agent_node in NODE_ORDER:
                if view_state["agent_status"][agent_node]["status"] != "失败":
                    _mark_node(view_state, agent_node, "完成", _node_detail(agent_node, view_state))
        else:
            _apply_error_statuses(view_state)
            _mark_incomplete_nodes_blocked(view_state)
    elif event_name == "error":
        error_text = str(payload.get("error", "后端流式任务失败"))
        failed_node = node if node in NODE_LABELS else "writer"
        _mark_node(view_state, failed_node, "失败", error_text)
        if error_text not in view_state["errors"]:
            view_state["errors"].append(error_text)

    return view_state


def main() -> None:
    """渲染 Streamlit 页面。"""
    st.set_page_config(page_title="Multi-Agent 研究助手", layout="wide")

    if "view_state" not in st.session_state:
        st.session_state["view_state"] = create_view_state()

    st.title("Multi-Agent 研究助手")
    with st.sidebar:
        api_base_url = st.text_input(
            "后端地址",
            value=get_settings().backend_url or DEFAULT_API_BASE_URL,
        )

    with st.form("research_form", clear_on_submit=False):
        topic = st.text_input(
            "研究主题",
            value=st.session_state["view_state"].get("topic", ""),
            placeholder="AI Agent 趋势",
        )
        submitted = st.form_submit_button("开始研究", type="primary")

    workspace = st.empty()
    if submitted:
        normalized_topic = topic.strip()
        if not normalized_topic:
            st.warning("请输入研究主题。")
        else:
            _run_research(
                topic=normalized_topic,
                api_base_url=api_base_url.strip() or DEFAULT_API_BASE_URL,
                workspace=workspace,
            )
    else:
        with workspace.container():
            _render_workspace(st.session_state["view_state"])


def _run_research(topic: str, api_base_url: str, workspace: Any) -> None:
    view_state = create_view_state(topic)
    st.session_state["view_state"] = view_state

    with workspace.container():
        _render_workspace(view_state)

    try:
        for event in subscribe_research(topic=topic, api_base_url=api_base_url):
            apply_event_to_view_state(event, view_state)
            st.session_state["view_state"] = view_state
            with workspace.container():
                _render_workspace(view_state)
            if event["event"] in {"complete", "error"}:
                break
    except requests.RequestException as exc:
        error_text = f"前端请求失败: {exc}"
        view_state["errors"].append(error_text)
        _mark_node(view_state, "planner", "失败", error_text)
        st.session_state["view_state"] = view_state
        with workspace.container():
            _render_workspace(view_state)


def _render_workspace(view_state: ViewState) -> None:
    st.progress(_overall_progress(view_state))
    _render_errors(view_state)
    _render_agent_status(view_state)
    _render_progress_details(view_state)
    _render_final_report(view_state)


def _render_agent_status(view_state: ViewState) -> None:
    columns = st.columns(3)
    for index, node in enumerate(NODE_ORDER):
        status = view_state["agent_status"][node]
        with columns[index]:
            st.subheader(NODE_LABELS[node])
            st.markdown(f"状态: **{status['status']}**")
            if status["detail"]:
                st.caption(status["detail"])


def _render_errors(view_state: ViewState) -> None:
    if not view_state["errors"]:
        return
    st.error("\n".join(view_state["errors"]))


def _render_progress_details(view_state: ViewState) -> None:
    planner_tab, researcher_tab, writer_tab = st.tabs(["Planner", "Researcher", "Writer"])

    with planner_tab:
        if view_state["sub_questions"]:
            for index, question in enumerate(view_state["sub_questions"], start=1):
                st.markdown(f"{index}. {question}")
        else:
            st.info("等待子问题。")

    with researcher_tab:
        st.metric("资料摘要", view_state["research_result_count"])
        st.metric("重试次数", view_state["retry_count"])
        if view_state["errors"]:
            st.error("\n".join(view_state["errors"]))

    with writer_tab:
        if view_state["final_report"]:
            st.success("报告已生成。")
            st.metric("报告字符数", len(view_state["final_report"]))
        else:
            st.info("等待报告。")


def _render_final_report(view_state: ViewState) -> None:
    if not view_state["final_report"]:
        return

    st.divider()
    st.markdown(view_state["final_report"])
    st.download_button(
        "下载 Markdown",
        data=view_state["final_report"],
        file_name=_report_filename(view_state["topic"]),
        mime="text/markdown",
        key=f"download_final_report_{len(view_state['events'])}",
    )


def _normalize_sse_line(raw_line: str | bytes) -> str:
    if isinstance(raw_line, bytes):
        raw_line = raw_line.decode("utf-8")
    return raw_line.rstrip("\r\n")


def _build_sse_event(event_name: str, data_lines: list[str]) -> SSEEvent:
    raw_data = "\n".join(data_lines)
    try:
        data: Any = json.loads(raw_data)
    except json.JSONDecodeError:
        data = raw_data
    return {"event": event_name, "data": data}


def _apply_state_summary(view_state: ViewState, state_summary: dict[str, Any]) -> None:
    view_state["topic"] = str(state_summary.get("topic", view_state["topic"]))
    view_state["sub_questions"] = _string_list(state_summary.get("sub_questions", []))
    view_state["research_result_count"] = int(state_summary.get("research_result_count", 0))
    view_state["final_report"] = str(state_summary.get("final_report", ""))
    view_state["errors"] = _string_list(state_summary.get("errors", []))
    view_state["retry_count"] = int(state_summary.get("retry_count", 0))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _mark_node(view_state: ViewState, node: str, status: str, detail: str) -> None:
    view_state["agent_status"][node] = {"status": status, "detail": detail}


def _display_status(status: str) -> str:
    if status == "failed":
        return "失败"
    if status == "warning":
        return "部分完成"
    if status == "empty":
        return "无结果"
    return "完成"


def _apply_error_statuses(view_state: ViewState) -> None:
    for error_text in view_state["errors"]:
        if error_text.startswith("Planner:"):
            _mark_node(view_state, "planner", "失败", error_text)
        elif error_text.startswith("Researcher:"):
            _mark_node(view_state, "researcher", "失败", error_text)
        elif error_text.startswith("Writer:"):
            _mark_node(view_state, "writer", "失败", error_text)


def _mark_incomplete_nodes_blocked(view_state: ViewState) -> None:
    for node in NODE_ORDER:
        if view_state["agent_status"][node]["status"] in {"等待", "运行中"}:
            _mark_node(view_state, node, "阻塞", "上游节点失败或没有可用输出")


def _mark_next_node_running(view_state: ViewState, node: str) -> None:
    current_index = NODE_ORDER.index(node)
    if current_index + 1 >= len(NODE_ORDER):
        return

    next_node = NODE_ORDER[current_index + 1]
    if view_state["agent_status"][next_node]["status"] == "等待":
        _mark_node(view_state, next_node, "运行中", "等待后端事件")


def _node_detail(node: str, view_state: ViewState) -> str:
    if node == "planner":
        return f"{len(view_state['sub_questions'])} 个子问题"
    if node == "researcher":
        return f"{view_state['research_result_count']} 个资料摘要"
    if node == "writer":
        return f"{len(view_state['final_report'])} 个字符"
    return ""


def _append_event_log(
    view_state: ViewState,
    event_name: str,
    node: str,
    payload: dict[str, Any],
) -> None:
    view_state["events"].append(
        {
            "event": event_name,
            "node": node,
            "status": str(payload.get("status", "")),
        }
    )
    view_state["events"] = view_state["events"][-30:]


def _overall_progress(view_state: ViewState) -> int:
    completed_count = sum(
        1
        for node in NODE_ORDER
        if view_state["agent_status"][node]["status"] == "完成"
    )
    return int((completed_count / len(NODE_ORDER)) * 100)


def _report_filename(topic: str) -> str:
    normalized_topic = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", topic).strip("_")
    return f"{(normalized_topic or 'research_report')[:40]}.md"


if __name__ == "__main__":
    main()
