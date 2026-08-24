from __future__ import annotations

import importlib

import pytest

from agents.researcher import researcher_node
from agents.state import ResearchState
from tools.web_search import SearchResult

researcher_module = importlib.import_module("agents.researcher")


def make_state(sub_questions: list[str] | None = None) -> ResearchState:
    return {
        "topic": "AI Agent 趋势",
        "sub_questions": (
            sub_questions if sub_questions is not None else ["问题 1", "问题 2", "问题 3"]
        ),
        "research_results": {},
        "final_report": "",
        "errors": [],
        "retry_count": 0,
    }


def make_search_result(question: str) -> SearchResult:
    return {
        "title": f"{question} title",
        "url": f"https://example.com/{question}",
        "snippet": f"{question} snippet",
        "source": "tavily",
    }


def test_researcher_returns_summary_per_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_web_search(query: str, max_results: int) -> list[SearchResult]:
        assert max_results == 3
        return [make_search_result(query)]

    def fake_summary(
        question: str,
        search_results: list[SearchResult],
        system_prompt: str,
        trace_id: str,
    ) -> str:
        assert "Researcher Agent" in system_prompt
        assert trace_id
        return f"{question} 的资料摘要"

    monkeypatch.setattr(researcher_module, "web_search", fake_web_search)
    monkeypatch.setattr(researcher_module, "_call_summary_model", fake_summary)

    result = researcher_node(make_state(["问题 A", "问题 B", "问题 C"]))

    assert result["errors"] == []
    assert result["research_results"]["问题 A"].startswith("问题 A 的资料摘要")
    assert "https://example.com/问题 A" in result["research_results"]["问题 A"]
    assert len(result["research_results"]) == 3


def test_researcher_continues_when_one_question_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_web_search(query: str, max_results: int) -> list[SearchResult]:
        if query == "失败问题":
            raise RuntimeError("search failed")
        return [make_search_result(query)]

    def fake_summary(
        question: str,
        search_results: list[SearchResult],
        system_prompt: str,
        trace_id: str,
    ) -> str:
        return f"{question} 摘要"

    monkeypatch.setattr(researcher_module, "web_search", fake_web_search)
    monkeypatch.setattr(researcher_module, "_call_summary_model", fake_summary)

    result = researcher_node(make_state(["成功问题", "失败问题", "另一个成功问题"]))

    assert set(result["research_results"]) == {"成功问题", "另一个成功问题"}
    assert len(result["errors"]) == 1
    assert "失败问题" in result["errors"][0]


def test_researcher_writes_error_for_empty_sub_questions() -> None:
    result = researcher_node(make_state([]))

    assert result["errors"] == ["Researcher: sub_questions must not be empty"]


def test_researcher_preserves_existing_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = make_state(["问题 A"])
    state["errors"] = ["Planner: previous error"]

    def fake_web_search(query: str, max_results: int) -> list[SearchResult]:
        raise RuntimeError("search failed")

    monkeypatch.setattr(researcher_module, "web_search", fake_web_search)

    result = researcher_node(state)

    assert result["errors"][0] == "Planner: previous error"
    assert result["errors"][1].startswith("Researcher: 问题 A")


def test_researcher_captures_prompt_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_prompt_load(name: str) -> str:
        raise RuntimeError("prompt unavailable")

    monkeypatch.setattr(researcher_module, "load_prompt", fail_prompt_load)

    result = researcher_node(make_state(["问题 A"]))

    assert result["errors"] == ["Researcher: prompt unavailable"]
