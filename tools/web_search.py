"""Researcher Agent 使用的网页搜索工具。"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SearchProvider = Literal["tavily", "duckduckgo"]


class SearchResult(TypedDict):
    title: str
    url: str
    snippet: str
    source: str


class WebSearchError(RuntimeError):
    """网页搜索失败时抛出,对应当前配置的搜索提供方。"""


def web_search(
    query: str,
    max_results: int = 3,
    provider: SearchProvider | None = None,
) -> list[SearchResult]:
    """搜索网页,返回标准化的结果摘要。"""
    normalized_query = query.strip()
    if not normalized_query:
        raise WebSearchError("search query must not be empty")
    if max_results < 1:
        raise WebSearchError("max_results must be greater than 0")

    selected_provider = _resolve_provider(provider)
    logger.info(
        "web_search input query=%r provider=%s max_results=%s",
        normalized_query,
        selected_provider,
        max_results,
    )

    if selected_provider == "tavily":
        results = _search_tavily(normalized_query, max_results)
    else:
        results = _search_duckduckgo(normalized_query, max_results)

    logger.info(
        "web_search output provider=%s results=%s",
        selected_provider,
        len(results),
    )
    return results


def _resolve_provider(provider: SearchProvider | None) -> SearchProvider:
    raw_provider = provider or os.getenv("SEARCH_PROVIDER", "tavily")
    normalized_provider = raw_provider.strip().lower()
    if normalized_provider in {"tavily", "duckduckgo"}:
        return normalized_provider  # type: ignore[return-value]
    raise WebSearchError(
        "unsupported search provider "
        f"{raw_provider!r}; expected 'tavily' or 'duckduckgo'"
    )


def _search_tavily(query: str, max_results: int) -> list[SearchResult]:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise WebSearchError("TAVILY_API_KEY is not configured")

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=max_results)
    except Exception as exc:  # pragma: no cover - SDK 具体异常类型会变化。
        raise WebSearchError(f"tavily search failed: {exc}") from exc

    raw_results = response.get("results", [])
    return _normalize_results(raw_results, source="tavily", max_results=max_results)


def _search_duckduckgo(query: str, max_results: int) -> list[SearchResult]:
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:  # pragma: no cover - 包的具体异常类型会变化。
        raise WebSearchError(f"duckduckgo search failed: {exc}") from exc

    return _normalize_results(
        raw_results,
        source="duckduckgo",
        max_results=max_results,
    )


def _normalize_results(
    raw_results: list[dict[str, Any]],
    source: str,
    max_results: int,
) -> list[SearchResult]:
    normalized_results: list[SearchResult] = []
    for raw_result in raw_results[:max_results]:
        title = str(raw_result.get("title") or "").strip()
        url = str(raw_result.get("url") or raw_result.get("href") or "").strip()
        snippet = str(
            raw_result.get("snippet")
            or raw_result.get("content")
            or raw_result.get("body")
            or ""
        ).strip()

        if not url:
            continue

        normalized_results.append(
            {
                "title": title or url,
                "url": url,
                "snippet": snippet,
                "source": source,
            }
        )

    if not normalized_results:
        raise WebSearchError(f"{source} returned no usable search results")
    return normalized_results
