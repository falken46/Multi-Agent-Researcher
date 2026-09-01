from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

import core.config as config_module
import eval.query_cache as query_cache_module
from eval.query_cache import (
    QueryCache,
    QueryCacheError,
    QueryCacheIntegrityError,
    QueryCacheMiss,
    build_cache_key,
)
from tools.web_search import SearchProvider, SearchResult


def _result(*, suffix: str = "one") -> SearchResult:
    return {
        "title": f"Result {suffix}",
        "url": f"https://example.com/{suffix}",
        "snippet": f"Snippet {suffix}",
        "source": "tavily",
    }


def test_record_then_replay_uses_normalized_query_without_api_key(
    runtime_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, SearchProvider | None]] = []

    def fail_if_api_key_is_read(self: object) -> str:
        raise AssertionError("query cache must not read project settings or API keys")

    def fake_fetcher(
        query: str,
        max_results: int = 3,
        provider: SearchProvider | None = None,
    ) -> list[SearchResult]:
        calls.append((query, max_results, provider))
        return [_result()]

    monkeypatch.setattr(
        config_module.Settings,
        "require_tavily_api_key",
        fail_if_api_key_is_read,
    )
    cache_dir = runtime_dir / "query-cache"
    recorder = QueryCache(cache_dir, snapshot_id="snapshot-1", mode="record")

    recorded = recorder.resolve(
        "  ＲＲＦ\n  融合  ",
        provider="tavily",
        max_results=3,
        fetcher=fake_fetcher,
    )
    replay = QueryCache(
        cache_dir,
        snapshot_id="snapshot-1",
        mode="replay-only",
    ).resolve(
        "RRF 融合",
        provider="tavily",
        max_results=3,
    )

    assert calls == [("RRF 融合", 3, "tavily")]
    assert recorded.from_cache is False
    assert replay.from_cache is True
    assert replay.results == recorded.results == [_result()]
    assert replay.cache_key == recorded.cache_key


def test_cache_key_changes_with_request_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = build_cache_key(
        "same query",
        provider="tavily",
        max_results=3,
        snapshot_id="snapshot-a",
    )

    assert base != build_cache_key(
        "same query",
        provider="duckduckgo",
        max_results=3,
        snapshot_id="snapshot-a",
    )
    assert base != build_cache_key(
        "same query",
        provider="tavily",
        max_results=2,
        snapshot_id="snapshot-a",
    )
    assert base != build_cache_key(
        "same query",
        provider="tavily",
        max_results=3,
        snapshot_id="snapshot-b",
    )

    monkeypatch.setattr(query_cache_module, "SCHEMA_VERSION", 2)
    assert base != build_cache_key(
        "same query",
        provider="tavily",
        max_results=3,
        snapshot_id="snapshot-a",
    )


def test_replay_only_miss_fails_closed_without_calling_fetcher(
    runtime_dir: Path,
) -> None:
    called = False

    def unexpected_fetcher(
        query: str,
        max_results: int = 3,
        provider: SearchProvider | None = None,
    ) -> list[SearchResult]:
        nonlocal called
        called = True
        return [_result()]

    cache = QueryCache(
        runtime_dir / "query-cache",
        snapshot_id="snapshot-miss",
        mode="replay-only",
    )

    with pytest.raises(QueryCacheMiss, match="replay-only"):
        cache.resolve(
            "missing query",
            provider="tavily",
            max_results=3,
            fetcher=unexpected_fetcher,
        )

    assert called is False
    assert list(runtime_dir.rglob("*.json")) == []


@pytest.mark.parametrize("failure_kind", ["exception", "empty"])
def test_record_does_not_cache_failures_or_empty_results(
    runtime_dir: Path,
    failure_kind: str,
) -> None:
    cache_dir = runtime_dir / failure_kind
    recorder = QueryCache(cache_dir, snapshot_id="snapshot-1", mode="record")

    def failed_fetcher(
        query: str,
        max_results: int = 3,
        provider: SearchProvider | None = None,
    ) -> list[SearchResult]:
        if failure_kind == "exception":
            raise RuntimeError("provider unavailable")
        return []

    expected_error = RuntimeError if failure_kind == "exception" else QueryCacheError
    with pytest.raises(expected_error):
        recorder.resolve(
            "uncached query",
            provider="tavily",
            max_results=3,
            fetcher=failed_fetcher,
        )

    assert list(cache_dir.glob("*.json")) == []
    replay = QueryCache(cache_dir, snapshot_id="snapshot-1", mode="replay-only")
    with pytest.raises(QueryCacheMiss):
        replay.resolve(
            "uncached query",
            provider="tavily",
            max_results=3,
        )


def test_replay_validates_request_fields(
    runtime_dir: Path,
) -> None:
    cache_dir = runtime_dir / "query-cache"
    recorder = QueryCache(cache_dir, snapshot_id="snapshot-1", mode="record")
    recorded = recorder.resolve(
        "tamper check",
        provider="tavily",
        max_results=3,
        fetcher=lambda query, max_results, provider: [_result()],
    )
    cache_path = cache_dir / f"{recorded.cache_key}.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["provider"] = "duckduckgo"
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    replay = QueryCache(cache_dir, snapshot_id="snapshot-1", mode="replay-only")
    with pytest.raises(QueryCacheIntegrityError, match="request field"):
        replay.resolve(
            "tamper check",
            provider="tavily",
            max_results=3,
        )


@pytest.mark.asyncio
async def test_concurrent_to_thread_records_each_key_once(
    runtime_dir: Path,
) -> None:
    cache_dir = runtime_dir / "query-cache"
    cache = QueryCache(cache_dir, snapshot_id="snapshot-1", mode="record")
    call_count = 0
    count_lock = threading.Lock()

    def slow_fetcher(
        query: str,
        max_results: int = 3,
        provider: SearchProvider | None = None,
    ) -> list[SearchResult]:
        nonlocal call_count
        with count_lock:
            call_count += 1
        time.sleep(0.03)
        return [_result()]

    searches = await asyncio.gather(
        *(
            asyncio.to_thread(
                cache.resolve,
                "shared query",
                provider="tavily",
                max_results=3,
                fetcher=slow_fetcher,
            )
            for _ in range(8)
        )
    )

    assert call_count == 1
    assert sum(not item.from_cache for item in searches) == 1
    assert len({item.cache_key for item in searches}) == 1
    assert len(list(cache_dir.glob("*.json"))) == 1
    assert list(cache_dir.glob("*.tmp")) == []
