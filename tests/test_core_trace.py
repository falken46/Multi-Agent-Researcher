from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from core.config import clear_settings_cache
from core.trace import emit, new_trace_id, read_events, summarize


def enable_trace(monkeypatch: pytest.MonkeyPatch, trace_dir: Path) -> None:
    monkeypatch.setenv("TRACE_ENABLED", "true")
    monkeypatch.setenv("TRACE_DIR", str(trace_dir))
    clear_settings_cache()


def test_emit_writes_jsonl_and_summarize_aggregates(
    monkeypatch: pytest.MonkeyPatch,
    runtime_dir: Path,
) -> None:
    trace_dir = runtime_dir
    enable_trace(monkeypatch, trace_dir)
    trace_id = new_trace_id()
    start_ts = 1_700_000_000.0

    emit(
        {
            "trace_id": trace_id,
            "ts": start_ts,
            "event": "task_start",
            "payload": {"topic": "test"},
        }
    )
    emit(
        {
            "trace_id": trace_id,
            "ts": start_ts + 0.5,
            "event": "llm_call",
            "node": "planner",
            "payload": {
                "success": True,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "latency_ms": 250.0,
                "cost": 0.001,
            },
        }
    )
    emit(
        {
            "trace_id": trace_id,
            "ts": start_ts + 2.0,
            "event": "task_end",
            "payload": {"status": "completed"},
        }
    )

    trace_files = list(trace_dir.glob(f"*/{trace_id}.jsonl"))
    assert len(trace_files) == 1
    lines = trace_files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[1])["event"] == "llm_call"

    summary = summarize(trace_id)
    assert summary["event_count"] == 3
    assert summary["total_latency_ms"] == pytest.approx(2000.0)
    assert summary["total_tokens"] == 120
    assert summary["llm_calls"] == 1
    assert summary["by_node"]["planner"]["calls"] == 1
    assert summary["total_cost"] == pytest.approx(0.001)
    assert [event["event"] for event in read_events(trace_id)] == [
        "task_start",
        "llm_call",
        "task_end",
    ]


def test_emit_is_thread_safe(
    monkeypatch: pytest.MonkeyPatch,
    runtime_dir: Path,
) -> None:
    enable_trace(monkeypatch, runtime_dir)
    trace_id = new_trace_id()

    def write_event(index: int) -> None:
        emit(
            {
                "trace_id": trace_id,
                "event": "tool_call",
                "node": "researcher",
                "payload": {"index": index},
            }
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write_event, range(50)))

    summary = summarize(trace_id)
    assert summary["event_count"] == 50


def test_emit_rejects_unsafe_trace_id(
    monkeypatch: pytest.MonkeyPatch,
    runtime_dir: Path,
) -> None:
    enable_trace(monkeypatch, runtime_dir)

    with pytest.raises(ValueError, match="trace_id"):
        emit(
            {
                "trace_id": "../escape",
                "event": "task_start",
                "payload": {},
            }
        )
