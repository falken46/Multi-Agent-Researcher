from __future__ import annotations

from typing import Any
import importlib

import pytest

from tools.web_fetch import WebFetchError, web_fetch
from tools.web_search import WebSearchError, web_search

web_fetch_module = importlib.import_module("tools.web_fetch")
web_search_module = importlib.import_module("tools.web_search")


def test_web_search_tavily_returns_structured_results(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_tavily(query: str, max_results: int) -> list[dict[str, str]]:
        assert query == "LangGraph tutorial"
        assert max_results == 2
        return [
            {
                "title": "LangGraph docs",
                "url": "https://langchain-ai.github.io/langgraph/",
                "snippet": "Build stateful agents.",
                "source": "tavily",
            },
            {
                "title": "LangGraph examples",
                "url": "https://example.com/langgraph",
                "snippet": "Examples.",
                "source": "tavily",
            },
        ]

    monkeypatch.setattr(web_search_module, "_search_tavily", fake_tavily)

    results = web_search("LangGraph tutorial", max_results=2, provider="tavily")

    assert len(results) == 2
    assert results[0]["title"] == "LangGraph docs"
    assert results[0]["url"].startswith("https://")
    assert results[0]["snippet"] == "Build stateful agents."
    assert results[0]["source"] == "tavily"


def test_web_search_tavily_failure_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_tavily(query: str, max_results: int) -> list[dict[str, str]]:
        raise WebSearchError("missing key")

    monkeypatch.setattr(web_search_module, "_search_tavily", fake_tavily)

    with pytest.raises(WebSearchError, match="missing key"):
        web_search("test query", provider="tavily")


def test_web_search_duckduckgo_is_explicit_backup(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_duckduckgo(query: str, max_results: int) -> list[dict[str, str]]:
        return [
            {
                "title": "Backup result",
                "url": "https://example.com/backup",
                "snippet": "Backup snippet",
                "source": "duckduckgo",
            }
        ]

    monkeypatch.setattr(web_search_module, "_search_duckduckgo", fake_duckduckgo)

    results = web_search("test query", provider="duckduckgo")

    assert results[0]["source"] == "duckduckgo"


def test_web_search_rejects_empty_query() -> None:
    with pytest.raises(WebSearchError, match="must not be empty"):
        web_search("   ")


def test_web_fetch_extracts_readable_text(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        text = """
        <html>
          <head><title>Example Domain</title><script>bad()</script></head>
          <body>
            <h1>Example Domain</h1>
            <p>This domain is for use in illustrative examples.</p>
          </body>
        </html>
        """

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, headers: dict[str, str], timeout: float) -> FakeResponse:
        assert url == "https://example.com"
        assert "User-Agent" in headers
        assert timeout == 10.0
        return FakeResponse()

    monkeypatch.setattr(web_fetch_module.requests, "get", fake_get)

    text = web_fetch("https://example.com")

    assert "Example Domain" in text
    assert "illustrative examples" in text
    assert "bad()" not in text


def test_web_fetch_rejects_invalid_url() -> None:
    with pytest.raises(WebFetchError, match="absolute http or https"):
        web_fetch("example.com")


def test_web_fetch_limits_output_length(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        text = "<html><body><p>abcdef</p></body></html>"

        def raise_for_status(self) -> None:
            return None

    def fake_get(*args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse()

    monkeypatch.setattr(web_fetch_module.requests, "get", fake_get)

    assert web_fetch("https://example.com", max_chars=3) == "abc"
