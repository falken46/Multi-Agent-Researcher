from __future__ import annotations

import importlib

import pytest

from agents.planner import planner_node
from agents.state import ResearchState

planner_module = importlib.import_module("agents.planner")


def make_state(topic: str = "AI Agent 趋势") -> ResearchState:
    return {
        "topic": topic,
        "sub_questions": [],
        "research_results": {},
        "final_report": "",
        "errors": [],
        "retry_count": 0,
    }


def test_planner_outputs_sub_questions_from_json_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call(topic: str, system_prompt: str) -> str:
        assert topic == "AI Agent 趋势"
        assert "Planner Agent" in system_prompt
        return """
        {
          "sub_questions": [
            "AI Agent 的架构趋势是什么?",
            "AI Agent 的企业应用场景有哪些?",
            "AI Agent 工程落地的挑战有哪些?"
          ]
        }
        """

    monkeypatch.setattr(planner_module, "_call_planner_model", fake_call)

    result = planner_node(make_state())

    assert result["sub_questions"] == [
        "AI Agent 的架构趋势是什么?",
        "AI Agent 的企业应用场景有哪些?",
        "AI Agent 工程落地的挑战有哪些?",
    ]


def test_planner_accepts_json_array_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_call(topic: str, system_prompt: str) -> str:
        return '["Q1", "Q2", "Q3"]'

    monkeypatch.setattr(planner_module, "_call_planner_model", fake_call)

    result = planner_node(make_state())

    assert result["sub_questions"] == ["Q1", "Q2", "Q3"]


def test_planner_writes_errors_for_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_call(topic: str, system_prompt: str) -> str:
        return "not json"

    monkeypatch.setattr(planner_module, "_call_planner_model", fake_call)

    result = planner_node(make_state())

    assert "errors" in result
    assert result["errors"]
    assert result["errors"][0].startswith("Planner:")


def test_planner_writes_errors_for_empty_topic() -> None:
    result = planner_node(make_state(topic="  "))

    assert "errors" in result
    assert "topic must not be empty" in result["errors"][0]


def test_planner_rejects_non_ascii_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-测试")

    with pytest.raises(planner_module.PlannerError, match="ASCII"):
        planner_module._call_planner_model("测试主题", "system prompt")
