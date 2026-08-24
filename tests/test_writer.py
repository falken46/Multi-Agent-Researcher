from __future__ import annotations

import importlib

import pytest

from agents.state import ResearchState
from agents.writer import build_writer_prompt, writer_node

writer_module = importlib.import_module("agents.writer")


def make_state() -> ResearchState:
    return {
        "topic": "AI Agent 趋势",
        "sub_questions": [
            "AI Agent 的架构趋势是什么?",
            "AI Agent 的企业应用有哪些?",
        ],
        "research_results": {
            "AI Agent 的架构趋势是什么?": "架构趋势摘要。\n\n来源:\n- https://example.com/architecture",
            "AI Agent 的企业应用有哪些?": "企业应用摘要。\n\n来源:\n- https://example.com/business",
        },
        "final_report": "",
        "errors": [],
        "retry_count": 0,
    }


def test_writer_outputs_final_report(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_call(user_prompt: str, system_prompt: str, trace_id: str) -> str:
        assert "AI Agent 趋势" in user_prompt
        assert "架构趋势摘要" in user_prompt
        assert "Writer Agent" in system_prompt
        assert trace_id
        return "# AI Agent 趋势\n\n## 摘要\n测试报告。"

    monkeypatch.setattr(writer_module, "_call_writer_model", fake_call)

    result = writer_node(make_state())

    assert result == {"final_report": "# AI Agent 趋势\n\n## 摘要\n测试报告。"}


def test_build_writer_prompt_includes_questions_results_and_errors() -> None:
    prompt = build_writer_prompt(
        topic="测试主题",
        sub_questions=["问题 A", "问题 B"],
        research_results={"问题 A": "资料 A"},
        errors=["Researcher: 问题 B | failed"],
    )

    assert "研究主题:\n测试主题" in prompt
    assert "问题: 问题 A" in prompt
    assert "资料 A" in prompt
    assert "问题: 问题 B" in prompt
    assert "资料不足或未检索到结果。" in prompt
    assert "Researcher: 问题 B | failed" in prompt


def test_writer_writes_error_when_research_results_missing() -> None:
    state = make_state()
    state["research_results"] = {}

    result = writer_node(state)

    assert "errors" in result
    assert "research_results must not be empty" in result["errors"][0]


def test_writer_preserves_existing_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    state = make_state()
    state["errors"] = ["Researcher: previous error"]

    def fake_call(user_prompt: str, system_prompt: str, trace_id: str) -> str:
        raise RuntimeError("model failed")

    monkeypatch.setattr(writer_module, "_call_writer_model", fake_call)

    result = writer_node(state)

    assert result["errors"][0] == "Researcher: previous error"
    assert result["errors"][1] == "Writer: model failed"


def test_writer_rejects_non_ascii_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-测试")

    with pytest.raises(writer_module.WriterError, match="ASCII"):
        writer_module._call_writer_model("user prompt", "system prompt", "trace-test")
