"""OpenTelemetry 导出测试（Phase 20）。

这里**没有 mock OTel**：用官方的 `InMemorySpanExporter` 走真实 SDK 代码路径，
把导出的 span 读回来断言。所以这些测试验证的是真的"span 树建对了"，
而不是"我们调了某个函数"。

代价是仍然验证不了**真实后端**（Laminar / Langfuse）能不能收下这些 span ——
那需要一个真实 endpoint。这条边界在 README 与 UPGRADE_V3 里都写明了。
"""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from core import otel
from core.config import Settings


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    monkeypatch.setenv("TRACE_ENABLED", "true")
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    # SimpleSpanProcessor 是同步导出，测试里不用等 batch 刷新
    provider.add_span_processor(SimpleSpanProcessor(memory))
    otel.configure_for_testing(provider)
    yield memory
    otel.reset()


def _event(name: str, *, node: str = "", trace_id: str = "t-1", **payload: Any) -> dict:
    return {"trace_id": trace_id, "event": name, "node": node, "payload": payload}


def _by_name(memory: InMemorySpanExporter) -> dict[str, Any]:
    return {span.name: span for span in memory.get_finished_spans()}


# ---------- 开关 ----------


def test_disabled_when_endpoint_is_empty() -> None:
    """未配置 endpoint 时完全不介入 —— 保住"无外部服务也能跑"。"""
    assert otel.is_enabled(Settings(_env_file=None, otel_endpoint="")) is False
    assert otel.is_enabled(Settings(_env_file=None, otel_endpoint="http://x/v1/traces"))


def test_parse_headers() -> None:
    parsed = otel._parse_headers("Authorization=Bearer abc, X-Foo=bar ,")
    assert parsed == {"Authorization": "Bearer abc", "X-Foo": "bar"}


# ---------- span 树 ----------


def test_task_events_produce_a_root_span(exporter: InMemorySpanExporter) -> None:
    otel.record(_event("task_start", topic="RRF"))
    otel.record(_event("task_end", status="completed"))

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "task.task_start"
    assert spans[0].attributes["trace_id"] == "t-1"
    assert spans[0].attributes["status"] == "completed"


def test_node_spans_are_children_of_the_task_span(
    exporter: InMemorySpanExporter,
) -> None:
    """节点 span 必须挂在任务 span 下，否则看板上是一堆平铺的孤立 span。"""
    otel.record(_event("task_start", topic="RRF"))
    otel.record(_event("node_start", node="planner"))
    otel.record(_event("node_end", node="planner"))
    otel.record(_event("task_end", status="completed"))

    spans = _by_name(exporter)
    root = spans["task.task_start"]
    planner = spans["node.planner"]

    assert planner.parent is not None
    assert planner.parent.span_id == root.context.span_id
    # 同一条 trace 下
    assert planner.context.trace_id == root.context.trace_id


def test_multiple_nodes_all_hang_under_the_root(
    exporter: InMemorySpanExporter,
) -> None:
    otel.record(_event("task_start"))
    for node in ("planner", "researcher", "critic", "writer"):
        otel.record(_event("node_start", node=node))
        otel.record(_event("node_end", node=node))
    otel.record(_event("task_end", status="completed"))

    spans = _by_name(exporter)
    root_id = spans["task.task_start"].context.span_id
    for node in ("planner", "researcher", "critic", "writer"):
        assert spans[f"node.{node}"].parent.span_id == root_id


# ---------- 无配对的事件 ----------


def test_point_events_become_span_events_not_lost(
    exporter: InMemorySpanExporter,
) -> None:
    """llm_call / fallback 这类没有配对的事件挂成 span event，不能丢。"""
    otel.record(_event("task_start"))
    otel.record(_event("node_start", node="researcher"))
    otel.record(_event("llm_call", node="researcher", total_tokens=120))
    otel.record(_event("fallback", node="researcher", threshold=0.35))
    otel.record(_event("node_end", node="researcher"))
    otel.record(_event("task_end", status="completed"))

    researcher = _by_name(exporter)["node.researcher"]
    names = [e.name for e in researcher.events]
    assert names == ["llm_call", "fallback"]
    assert researcher.events[0].attributes["total_tokens"] == 120


def test_event_without_node_falls_back_to_root(
    exporter: InMemorySpanExporter,
) -> None:
    otel.record(_event("task_start"))
    otel.record(_event("revision", count=1))
    otel.record(_event("task_end", status="completed"))

    root = _by_name(exporter)["task.task_start"]
    assert [e.name for e in root.events] == ["revision"]


def test_error_event_marks_the_span(exporter: InMemorySpanExporter) -> None:
    otel.record(_event("task_start"))
    otel.record(_event("node_start", node="writer"))
    otel.record(_event("error", node="writer", message="boom"))
    otel.record(_event("node_end", node="writer"))
    otel.record(_event("task_end", status="failed"))

    writer = _by_name(exporter)["node.writer"]
    assert writer.attributes["error"] is True


# ---------- 异常路径 ----------


def test_unclosed_node_span_is_closed_by_task_end(
    exporter: InMemorySpanExporter,
) -> None:
    """节点抛异常时没有 node_end；不兜底的话这个 span 永远不结束，看板上像卡住。"""
    otel.record(_event("task_start"))
    otel.record(_event("node_start", node="researcher"))
    otel.record(_event("task_end", status="failed"))

    spans = _by_name(exporter)
    assert "node.researcher" in spans
    assert spans["node.researcher"].attributes["unclosed"] is True


def test_record_never_raises(exporter: InMemorySpanExporter) -> None:
    """可观测出问题不能拖垮被观测的系统。"""
    otel.record({"event": "task_start"})  # 没有 trace_id
    otel.record(_event("node_end", node="never-started"))  # 结束一个没开的 span
    otel.record(_event("task_end"))  # 结束一个没开的任务
    # 走到这里就说明没抛异常


def test_non_scalar_payload_is_stringified(exporter: InMemorySpanExporter) -> None:
    """OTel 属性只接受标量与同类标量列表，其余必须转字符串，否则 SDK 会告警丢弃。"""
    otel.record(_event("task_start", nested={"a": 1}, tags=["x", "y"], n=3))
    otel.record(_event("task_end"))

    root = _by_name(exporter)["task.task_start"]
    assert root.attributes["nested"] == "{'a': 1}"
    assert list(root.attributes["tags"]) == ["x", "y"]
    assert root.attributes["n"] == 3


def test_two_traces_do_not_mix(exporter: InMemorySpanExporter) -> None:
    otel.record(_event("task_start", trace_id="a"))
    otel.record(_event("task_start", trace_id="b"))
    otel.record(_event("node_start", node="planner", trace_id="a"))
    otel.record(_event("node_end", node="planner", trace_id="a"))
    otel.record(_event("task_end", trace_id="a"))
    otel.record(_event("task_end", trace_id="b"))

    planner = [s for s in exporter.get_finished_spans() if s.name == "node.planner"][0]
    roots = {
        s.attributes["trace_id"]: s
        for s in exporter.get_finished_spans()
        if s.name == "task.task_start"
    }
    assert planner.parent.span_id == roots["a"].context.span_id
    assert planner.parent.span_id != roots["b"].context.span_id
