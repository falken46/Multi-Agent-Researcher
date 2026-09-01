"""Phase 13 对照实验使用的 query-keyed Web 搜索快照缓存。

缓存只保存显式传入 fetcher 的标准化搜索结果，不读取项目配置或 API Key。
正式实验先用 ``record`` 模式预热，再用 ``replay-only`` 模式只读回放；
回放缺失或缓存损坏时均失败关闭，避免不同实验组静默拿到不同证据。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import unicodedata
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol, cast

from tools.web_search import SearchProvider, SearchResult

SCHEMA_VERSION = 1
CacheMode = Literal["record", "replay-only"]

_path_locks: dict[Path, threading.Lock] = {}
_path_locks_guard = threading.Lock()


class QueryCacheError(RuntimeError):
    """Query cache 无法安全返回结果。"""


class QueryCacheMiss(QueryCacheError):
    """Replay-only 模式下没有对应的预热结果。"""


class QueryCacheIntegrityError(QueryCacheError):
    """缓存内容与请求或结果哈希不一致。"""


class SearchFetcher(Protocol):
    """与 :func:`tools.web_search.web_search` 兼容的最小调用协议。"""

    def __call__(
        self,
        query: str,
        max_results: int = 3,
        provider: SearchProvider | None = None,
    ) -> list[SearchResult]: ...


@dataclass(frozen=True)
class CachedSearch:
    """一次缓存解析结果及其可审计标识。"""

    results: list[SearchResult]
    cache_key: str
    from_cache: bool


class QueryCache:
    """按查询请求写入不可变 JSON 文件，并支持严格只读回放。"""

    def __init__(
        self,
        directory: Path | str,
        *,
        snapshot_id: str,
        mode: CacheMode,
    ) -> None:
        normalized_snapshot_id = snapshot_id.strip()
        if not normalized_snapshot_id:
            raise ValueError("snapshot_id must not be empty")
        if mode not in {"record", "replay-only"}:
            raise ValueError("mode must be 'record' or 'replay-only'")
        self._directory = Path(directory)
        self.snapshot_id = normalized_snapshot_id
        self.mode = mode

    def resolve(
        self,
        query: str,
        *,
        provider: SearchProvider,
        max_results: int,
        fetcher: SearchFetcher | None = None,
    ) -> CachedSearch:
        """读取缓存；record miss 时调用 fetcher，replay miss 时明确失败。"""
        request = _request_payload(
            query=query,
            provider=provider,
            max_results=max_results,
            snapshot_id=self.snapshot_id,
        )
        cache_key = _hash_json(request)
        path = self._directory / f"{cache_key}.json"

        with _lock_for_path(path):
            if path.is_file():
                results = _read_entry(
                    path,
                    expected_request=request,
                )
                return CachedSearch(
                    results=results,
                    cache_key=cache_key,
                    from_cache=True,
                )

            if self.mode == "replay-only":
                raise QueryCacheMiss(
                    "query cache miss in replay-only mode: "
                    f"snapshot={self.snapshot_id!r} key={cache_key}"
                )
            if fetcher is None:
                raise QueryCacheError("record mode requires a fetcher on cache miss")

            normalized_query = cast(str, request["normalized_query"])
            normalized_provider = cast(SearchProvider, request["provider"])
            raw_results = fetcher(
                normalized_query,
                max_results,
                normalized_provider,
            )
            results = _normalize_results(raw_results, max_results=max_results)
            entry = {
                **request,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "results": results,
            }
            _atomic_write_json(path, entry)
            return CachedSearch(
                results=_copy_results(results),
                cache_key=cache_key,
                from_cache=False,
            )


def normalize_query(query: str) -> str:
    """做保守的 Unicode 与空白规范化，不删除可能影响语义的标点。"""
    normalized = " ".join(unicodedata.normalize("NFKC", query).split())
    if not normalized:
        raise ValueError("query must not be empty")
    return normalized


def build_cache_key(
    query: str,
    *,
    provider: SearchProvider,
    max_results: int,
    snapshot_id: str,
) -> str:
    """返回包含完整请求身份的稳定 SHA-256 key。"""
    return _hash_json(
        _request_payload(
            query=query,
            provider=provider,
            max_results=max_results,
            snapshot_id=snapshot_id,
        )
    )


def _request_payload(
    *,
    query: str,
    provider: SearchProvider,
    max_results: int,
    snapshot_id: str,
) -> dict[str, object]:
    normalized_snapshot_id = snapshot_id.strip()
    if not normalized_snapshot_id:
        raise ValueError("snapshot_id must not be empty")
    normalized_provider = str(provider).strip().lower()
    if normalized_provider not in {"tavily", "duckduckgo"}:
        raise ValueError("provider must be 'tavily' or 'duckduckgo'")
    if (
        isinstance(max_results, bool)
        or not isinstance(max_results, int)
        or max_results < 1
    ):
        raise ValueError("max_results must be greater than zero")
    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": normalized_snapshot_id,
        "normalized_query": normalize_query(query),
        "provider": normalized_provider,
        "max_results": max_results,
    }


def _read_entry(
    path: Path,
    *,
    expected_request: Mapping[str, object],
) -> list[SearchResult]:
    try:
        raw_entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QueryCacheIntegrityError(
            f"failed to read query cache entry: {path}"
        ) from exc
    if not isinstance(raw_entry, dict):
        raise QueryCacheIntegrityError("query cache entry must be a JSON object")

    for field, expected_value in expected_request.items():
        if raw_entry.get(field) != expected_value:
            raise QueryCacheIntegrityError(
                f"query cache request field mismatch: {field}"
            )

    max_results = cast(int, expected_request["max_results"])
    try:
        results = _normalize_results(
            raw_entry.get("results"),
            max_results=max_results,
        )
    except QueryCacheError as exc:
        raise QueryCacheIntegrityError(str(exc)) from exc
    return results


def _normalize_results(
    raw_results: object,
    *,
    max_results: int,
) -> list[SearchResult]:
    if not isinstance(raw_results, list) or not raw_results:
        raise QueryCacheError("successful search results must be a non-empty list")
    if len(raw_results) > max_results:
        raise QueryCacheError("search result count exceeds max_results")

    results: list[SearchResult] = []
    for index, raw_result in enumerate(raw_results):
        if not isinstance(raw_result, Mapping):
            raise QueryCacheError(f"search result {index} must be an object")
        title = _required_string(raw_result, "title", index=index)
        url = _required_string(raw_result, "url", index=index)
        snippet = _string(raw_result, "snippet", index=index)
        source = _required_string(raw_result, "source", index=index)
        results.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "source": source,
            }
        )
    return results


def _required_string(
    item: Mapping[object, object],
    field: str,
    *,
    index: int,
) -> str:
    value = _string(item, field, index=index)
    if not value:
        raise QueryCacheError(
            f"search result {index} field {field!r} must not be empty"
        )
    return value


def _string(
    item: Mapping[object, object],
    field: str,
    *,
    index: int,
) -> str:
    value = item.get(field)
    if not isinstance(value, str):
        raise QueryCacheError(
            f"search result {index} field {field!r} must be a string"
        )
    return value.strip()


def _copy_results(results: list[SearchResult]) -> list[SearchResult]:
    return [
        {
            "title": item["title"],
            "url": item["url"],
            "snippet": item["snippet"],
            "source": item["source"],
        }
        for item in results
    ]


def _hash_json(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _lock_for_path(path: Path) -> threading.Lock:
    resolved_path = path.resolve()
    with _path_locks_guard:
        return _path_locks.setdefault(resolved_path, threading.Lock())


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("x", encoding="utf-8", newline="\n") as cache_file:
            json.dump(
                payload,
                cache_file,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            cache_file.write("\n")
            cache_file.flush()
            os.fsync(cache_file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


__all__ = [
    "CacheMode",
    "CachedSearch",
    "QueryCache",
    "QueryCacheError",
    "QueryCacheIntegrityError",
    "QueryCacheMiss",
    "SCHEMA_VERSION",
    "SearchFetcher",
    "build_cache_key",
    "normalize_query",
]
