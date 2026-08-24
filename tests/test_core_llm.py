from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import core.llm as llm_module
from core.config import clear_settings_cache
from core.llm import LLMCallError, achat, chat
from core.trace import summarize


class FakeCompletions:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeAsyncCompletions(FakeCompletions):
    async def create(self, **kwargs: Any) -> Any:
        return super().create(**kwargs)


class FakeStatusError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def make_response(content: str = "ok") -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=25,
            total_tokens=125,
            prompt_cache_hit_tokens=40,
            prompt_cache_miss_tokens=60,
        ),
    )


def configure_llm(
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_retry: int = 3,
    trace_dir: Path | None = None,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MAX_RETRY", str(max_retry))
    if trace_dir is not None:
        monkeypatch.setenv("TRACE_ENABLED", "true")
        monkeypatch.setenv("TRACE_DIR", str(trace_dir))
    clear_settings_cache()


def test_chat_returns_content_usage_cost_and_json_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_llm(monkeypatch)
    completions = FakeCompletions([make_response(" structured ")])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(llm_module, "_create_sync_client", lambda settings: client)

    result = chat(
        [{"role": "user", "content": "hello"}],
        node="planner",
        trace_id="trace-chat",
        json_mode=True,
    )

    assert result.content == "structured"
    assert result.usage.prompt_tokens == 100
    assert result.usage.cache_hit_tokens == 40
    assert result.cost == pytest.approx(0.0001108)
    assert completions.calls[0]["response_format"] == {"type": "json_object"}


def test_chat_retries_timeout_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_llm(monkeypatch, max_retry=2)
    completions = FakeCompletions([TimeoutError("slow"), make_response()])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    delays: list[float] = []
    monkeypatch.setattr(llm_module, "_create_sync_client", lambda settings: client)
    monkeypatch.setattr(llm_module.time, "sleep", delays.append)

    result = chat(
        [{"role": "user", "content": "hello"}],
        node="researcher",
        trace_id="trace-retry",
    )

    assert result.content == "ok"
    assert len(completions.calls) == 2
    assert delays == [1.0]


def test_chat_does_not_retry_4xx(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_llm(monkeypatch, max_retry=3)
    completions = FakeCompletions([FakeStatusError(400)])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(llm_module, "_create_sync_client", lambda settings: client)

    with pytest.raises(LLMCallError, match="1 attempt"):
        chat(
            [{"role": "user", "content": "hello"}],
            node="writer",
            trace_id="trace-4xx",
        )

    assert len(completions.calls) == 1


def test_chat_retries_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_llm(monkeypatch, max_retry=1)
    completions = FakeCompletions([FakeStatusError(503), make_response()])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(llm_module, "_create_sync_client", lambda settings: client)
    monkeypatch.setattr(llm_module.time, "sleep", lambda delay: None)

    assert chat(
        [{"role": "user", "content": "hello"}],
        node="writer",
        trace_id="trace-5xx",
    ).content == "ok"
    assert len(completions.calls) == 2


@pytest.mark.asyncio
async def test_achat_uses_async_client(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_llm(monkeypatch)
    completions = FakeAsyncCompletions([make_response("async ok")])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(llm_module, "_create_async_client", lambda settings: client)

    result = await achat(
        [{"role": "user", "content": "hello"}],
        node="researcher",
        trace_id="trace-async",
    )

    assert result.content == "async ok"
    assert len(completions.calls) == 1


def test_chat_emits_trace_with_usage(
    monkeypatch: pytest.MonkeyPatch,
    runtime_dir: Path,
) -> None:
    configure_llm(
        monkeypatch,
        trace_dir=runtime_dir,
    )
    completions = FakeCompletions([make_response()])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(llm_module, "_create_sync_client", lambda settings: client)

    chat(
        [{"role": "user", "content": "hello"}],
        node="planner",
        trace_id="trace-integration",
    )

    summary = summarize("trace-integration")
    assert summary["llm_calls"] == 1
    assert summary["total_tokens"] == 125
    assert summary["by_node"]["planner"]["calls"] == 1
