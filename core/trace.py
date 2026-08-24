"""轻量 JSONL 调用链记录与聚合。"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, TypedDict

from core.config import get_settings

logger = logging.getLogger(__name__)

TRACE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_path_locks: defaultdict[Path, threading.Lock] = defaultdict(threading.Lock)
_path_locks_guard = threading.Lock()


class TraceEvent(TypedDict, total=False):
    trace_id: str
    ts: float
    event: str
    node: str
    payload: dict[str, Any]


def new_trace_id() -> str:
    """生成一次任务使用的全链路追踪 ID。"""
    return str(uuid.uuid4())


def emit(event: Mapping[str, Any]) -> None:
    """将单个结构化事件追加写入对应 trace JSONL。"""
    settings = get_settings()
    if not settings.trace_enabled:
        return

    normalized_event = _normalize_event(event)
    trace_path = _trace_path(
        trace_id=normalized_event["trace_id"],
        ts=normalized_event["ts"],
        trace_dir=settings.trace_dir,
    )
    serialized = json.dumps(normalized_event, ensure_ascii=False, separators=(",", ":"))

    try:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        lock = _lock_for_path(trace_path)
        with lock, trace_path.open("a", encoding="utf-8", newline="\n") as trace_file:
            trace_file.write(serialized + "\n")
        logger.info(
            "trace emit trace_id=%s event=%s node=%s",
            normalized_event["trace_id"],
            normalized_event["event"],
            normalized_event.get("node", ""),
        )
    except OSError as exc:
        logger.error(
            "trace write failed trace_id=%s event=%s error=%s",
            normalized_event["trace_id"],
            normalized_event["event"],
            exc,
        )


def summarize(trace_id: str) -> dict[str, Any]:
    """从 trace 事件聚合 token、成本、耗时和关键行为指标。"""
    _validate_trace_id(trace_id)
    events = _read_events(trace_id, get_settings().trace_dir)
    by_node: defaultdict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "latency_ms": 0.0,
            "tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost": 0.0,
            "calls": 0,
        }
    )
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    llm_calls = 0
    fallback_count = 0
    revision_count = 0
    errors: list[dict[str, Any]] = []
    task_start_ts: float | None = None
    task_end_ts: float | None = None
    node_latency_ms = 0.0

    for event in events:
        event_name = str(event.get("event", ""))
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}

        if event_name == "task_start":
            task_start_ts = _as_float(event.get("ts"))
        elif event_name == "task_end":
            task_end_ts = _as_float(event.get("ts"))
        elif event_name == "node_end":
            node_latency_ms += _as_float(payload.get("latency_ms"))
        elif event_name == "fallback":
            fallback_count += 1
        elif event_name == "revision":
            revision_count += 1
        elif event_name == "error":
            errors.append(
                {
                    "node": event.get("node", ""),
                    "type": payload.get("type", ""),
                    "message": payload.get("message", ""),
                }
            )

        if event_name != "llm_call" or payload.get("success", True) is False:
            continue

        prompt_tokens = _as_int(payload.get("prompt_tokens"))
        completion_tokens = _as_int(payload.get("completion_tokens"))
        latency_ms = _as_float(payload.get("latency_ms"))
        cost = _as_float(payload.get("cost"))
        node = str(event.get("node", "unknown")) or "unknown"
        node_data = by_node[node]
        node_data["latency_ms"] += latency_ms
        node_data["tokens"] += prompt_tokens + completion_tokens
        node_data["prompt_tokens"] += prompt_tokens
        node_data["completion_tokens"] += completion_tokens
        node_data["cost"] += cost
        node_data["calls"] += 1
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_cost += cost
        llm_calls += 1

    if task_start_ts is not None and task_end_ts is not None:
        total_latency_ms = max(0.0, (task_end_ts - task_start_ts) * 1000)
    else:
        total_latency_ms = node_latency_ms

    settings = get_settings()
    return {
        "trace_id": trace_id,
        "event_count": len(events),
        "total_latency_ms": round(total_latency_ms, 3),
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_cost": round(total_cost, 10),
        "currency": settings.model_pricing_currency,
        "pricing_version": settings.model_pricing_version,
        "llm_calls": llm_calls,
        "by_node": {
            node: {
                key: round(value, 10) if isinstance(value, float) else value
                for key, value in values.items()
            }
            for node, values in sorted(by_node.items())
        },
        "fallback_count": fallback_count,
        "revision_count": revision_count,
        "errors": errors,
    }


def _normalize_event(event: Mapping[str, Any]) -> dict[str, Any]:
    trace_id = str(event.get("trace_id", "")).strip()
    _validate_trace_id(trace_id)
    event_name = str(event.get("event", "")).strip()
    if not event_name:
        raise ValueError("trace event name must not be empty")
    payload = event.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("trace event payload must be a dictionary")

    normalized: dict[str, Any] = {
        "trace_id": trace_id,
        "ts": _as_float(event.get("ts")) or time.time(),
        "event": event_name,
        "payload": payload,
    }
    node = str(event.get("node", "")).strip()
    if node:
        normalized["node"] = node
    return normalized


def _validate_trace_id(trace_id: str) -> None:
    if not TRACE_ID_PATTERN.fullmatch(trace_id):
        raise ValueError("trace_id contains unsupported characters")


def _trace_path(trace_id: str, ts: float, trace_dir: Path) -> Path:
    date_part = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    return Path(trace_dir) / date_part / f"{trace_id}.jsonl"


def _lock_for_path(path: Path) -> threading.Lock:
    with _path_locks_guard:
        return _path_locks[path]


def _read_events(trace_id: str, trace_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(Path(trace_dir).glob(f"*/{trace_id}.jsonl"))
    events: list[dict[str, Any]] = []
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as trace_file:
                for line_number, line in enumerate(trace_file, start=1):
                    stripped_line = line.strip()
                    if not stripped_line:
                        continue
                    try:
                        item = json.loads(stripped_line)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "trace line skipped path=%s line=%s error=%s",
                            path,
                            line_number,
                            exc,
                        )
                        continue
                    if isinstance(item, dict):
                        events.append(item)
        except OSError as exc:
            logger.warning("trace read skipped path=%s error=%s", path, exc)
    return sorted(events, key=lambda item: _as_float(item.get("ts")))


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = ["TraceEvent", "emit", "new_trace_id", "summarize"]

