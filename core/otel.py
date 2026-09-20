"""把 trace 事件流导出成 OpenTelemetry span（Phase 20）。

## 为什么是 OTel，而不是绑定某家 SDK

Laminar 是 OpenTelemetry 原生的，Langfuse 也接受 OTLP。用标准协议埋点之后，
换后端只改 ``OTEL_ENDPOINT`` 与 ``OTEL_HEADERS``，业务代码一行不动 ——
和 `rag/vectorstore.py` 用窄接口隔离向量库供应商是同一套思路。

## 事件流 → span 树的映射

`core/trace.py` 的模型是**追加写的事件流**，OTel 的模型是**有起止和父子关系的 span**。
两者不是一回事，但本项目的事件天然成对，可以干净地映射：

```
task_start / task_resume ──┐
                           ├─→ 根 span（一次任务）
task_end ──────────────────┘
                              │
    node_start ──┐            ├─ 子 span（planner / researcher / critic / writer）
    node_end ────┘            │
                              │
    llm_call / retrieval /    └─ span event（挂在当前打开的 span 上）
    fallback / revision /
    error / web_cache
```

**没有配对的事件不丢**：它们成为最近一个打开的 span 上的 event，
而不是被扔掉或者变成一堆零长度的孤立 span。

## 三条硬约束

1. **JSONL 永远是主路径。** 本模块是叠加的一层，未配置 endpoint 时完全不介入。
   "没有任何外部服务也能完整跑"是 v2 的既有优点，不能因为加了可观测就丢掉。
2. **本模块的任何异常都不能影响主流程。** 上报失败只记日志。
3. **不改 `core/trace.py` 的事件模型。** 那是 v2 已验证的东西，只在落地层叠加。
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Mapping

from core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# 会开启一个 span 的事件（值为 span 名的前缀）
_ROOT_START_EVENTS = {"task_start", "task_resume"}
_ROOT_END_EVENT = "task_end"
_NODE_START_EVENT = "node_start"
_NODE_END_EVENT = "node_end"

_lock = threading.Lock()
_provider: Any = None
_tracer: Any = None
_configured_for: tuple[str, str] | None = None
# trace_id -> {"root": span, "nodes": {node_name: span}}
_open_spans: dict[str, dict[str, Any]] = {}


def is_enabled(settings: Settings | None = None) -> bool:
    current = settings or get_settings()
    return bool(current.otel_endpoint.strip())


def _parse_headers(raw: str) -> dict[str, str]:
    """解析 ``Authorization=Bearer xxx,X-Foo=bar`` 形式的头。"""
    headers: dict[str, str] = {}
    for item in raw.split(","):
        entry = item.strip()
        if not entry or "=" not in entry:
            continue
        key, _, value = entry.partition("=")
        key = key.strip()
        if key:
            headers[key] = value.strip()
    return headers


def _get_tracer(settings: Settings) -> Any:
    """按 endpoint 懒初始化 TracerProvider。配置变了会重建。"""
    global _provider, _tracer, _configured_for

    endpoint = settings.otel_endpoint.strip()
    signature = (endpoint, settings.otel_service_name)
    if _tracer is not None and _configured_for == signature:
        return _tracer

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.otel_service_name})
    )
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=endpoint,
                headers=_parse_headers(settings.otel_headers.get_secret_value()),
                timeout=int(settings.otel_timeout),
            )
        )
    )
    _provider = provider
    _tracer = provider.get_tracer("deepresearch.trace")
    _configured_for = signature
    logger.info("otel exporter configured endpoint=%s", endpoint)
    return _tracer


def configure_for_testing(tracer_provider: Any) -> None:
    """测试专用：注入一个自带 exporter 的 provider。

    有了它就能在**没有任何服务器**的情况下验证 span 树是否正确 ——
    配 ``InMemorySpanExporter`` 就能把导出的 span 直接读回来断言。
    这不是 mock：走的是真实的 OTel SDK 代码路径。
    """
    global _provider, _tracer, _configured_for
    with _lock:
        _provider = tracer_provider
        _tracer = tracer_provider.get_tracer("deepresearch.trace")
        _configured_for = ("__testing__", "__testing__")
        _open_spans.clear()


def reset() -> None:
    """清掉 provider 与未关闭的 span。测试之间必须调用。"""
    global _provider, _tracer, _configured_for
    with _lock:
        for entry in _open_spans.values():
            for span in [entry.get("root"), *entry.get("nodes", {}).values()]:
                if span is not None:
                    try:
                        span.end()
                    except Exception:  # noqa: BLE001 - 清理路径不允许抛
                        pass
        _open_spans.clear()
        _provider = None
        _tracer = None
        _configured_for = None


def record(event: Mapping[str, Any], *, settings: Settings | None = None) -> None:
    """把一个 trace 事件反映到 OTel span 上。永不抛异常。"""
    current = settings or get_settings()
    testing = _configured_for == ("__testing__", "__testing__")
    if not testing and not is_enabled(current):
        return

    try:
        with _lock:
            tracer = _tracer if testing else _get_tracer(current)
            if tracer is None:
                return
            _apply(tracer, event)
    except Exception:  # noqa: BLE001
        # 可观测出问题不能拖垮被观测的系统 —— 这是可观测性的基本纪律
        logger.exception("otel record failed event=%s", event.get("event"))


def _apply(tracer: Any, event: Mapping[str, Any]) -> None:
    trace_id = str(event.get("trace_id") or "")
    if not trace_id:
        return
    name = str(event.get("event") or "")
    node = str(event.get("node") or "")
    payload = event.get("payload") or {}

    entry = _open_spans.setdefault(trace_id, {"root": None, "nodes": {}})

    if name in _ROOT_START_EVENTS:
        span = tracer.start_span(f"task.{name}")
        _set_attributes(span, trace_id=trace_id, node=node, payload=payload)
        entry["root"] = span
        return

    if name == _ROOT_END_EVENT:
        _close_remaining_nodes(entry)
        span = entry.get("root")
        if span is not None:
            _set_attributes(span, trace_id=trace_id, node=node, payload=payload)
            span.end()
        _open_spans.pop(trace_id, None)
        return

    if name == _NODE_START_EVENT and node:
        from opentelemetry import trace as otel_trace

        root = entry.get("root")
        # 显式指定父 span，而不是依赖 OTel 的隐式 context —— 本项目的事件
        # 跨线程、跨 async 任务发出，隐式 context 传不过去。
        context = otel_trace.set_span_in_context(root) if root is not None else None
        span = tracer.start_span(f"node.{node}", context=context)
        _set_attributes(span, trace_id=trace_id, node=node, payload=payload)
        entry["nodes"][node] = span
        return

    if name == _NODE_END_EVENT and node:
        span = entry["nodes"].pop(node, None)
        if span is not None:
            _set_attributes(span, trace_id=trace_id, node=node, payload=payload)
            span.end()
        return

    # 其余事件（llm_call / retrieval / fallback / revision / error / web_cache）
    # 挂到当前最相关的 span 上：优先该节点的 span，其次根 span。
    target = entry["nodes"].get(node) or entry.get("root")
    if target is None:
        return
    target.add_event(name, attributes=_flatten(payload))
    if name == "error":
        target.set_attribute("error", True)


def _close_remaining_nodes(entry: dict[str, Any]) -> None:
    """任务结束时兜底关掉还开着的节点 span。

    正常流程每个 node_start 都有 node_end，但节点抛异常时可能没有。
    不兜底的话这些 span 永远不结束，在看板上表现为"卡住"。
    """
    for node, span in list(entry.get("nodes", {}).items()):
        try:
            span.set_attribute("unclosed", True)
            span.end()
        except Exception:  # noqa: BLE001
            pass
        entry["nodes"].pop(node, None)


def _set_attributes(
    span: Any, *, trace_id: str, node: str, payload: Mapping[str, Any]
) -> None:
    span.set_attribute("trace_id", trace_id)
    if node:
        span.set_attribute("node", node)
    for key, value in _flatten(payload).items():
        span.set_attribute(key, value)


def _flatten(payload: Mapping[str, Any]) -> dict[str, Any]:
    """OTel 属性只接受标量与同类标量列表，其余转成字符串。"""
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        name = str(key)
        if isinstance(value, (str, bool, int, float)):
            flat[name] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(item, str) for item in value
        ):
            flat[name] = list(value)
        elif value is not None:
            flat[name] = str(value)
    return flat


__all__ = ["configure_for_testing", "is_enabled", "record", "reset"]
