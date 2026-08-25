from __future__ import annotations

import importlib

import pytest

from agents.graph import create_initial_state
from agents.state import ResearchState
from agents.writer import build_writer_prompt, writer_node

writer_module = importlib.import_module("agents.writer")


def make_state() -> ResearchState:
    state = create_initial_state("AI Agent 趋势")
    state["sub_questions"] = [
        "AI Agent 的架构趋势是什么?",
        "AI Agent 的企业应用有哪些?",
    ]
    state["research_results"] = {
        "AI Agent 的架构趋势是什么?": "架构趋势摘要。",
        "AI Agent 的企业应用有哪些?": "企业应用摘要。",
    }
    state["citations"] = {
        "AI Agent 的架构趋势是什么?": [
            {
                "source": "kb/architecture.md",
                "origin": "kb",
                "snippet": "架构证据",
            }
        ],
        "AI Agent 的企业应用有哪些?": [
            {
                "source": "https://example.com/business",
                "origin": "web",
                "snippet": "业务证据",
            }
        ],
    }
    return state


def test_writer_outputs_final_report(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_call(user_prompt: str, system_prompt: str, trace_id: str) -> str:
        assert "AI Agent 趋势" in user_prompt
        assert "架构趋势摘要" in user_prompt
        assert "[1] (kb) kb/architecture.md" in user_prompt
        assert "[2] (web) https://example.com/business" in user_prompt
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


def test_writer_consumes_directed_revision_results_and_structured_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = make_state()
    state["research_results"]["补充 2026 行业风险"] = "定向补查后的新证据"
    state["citations"]["补充 2026 行业风险"] = [
        {
            "source": "https://example.com/risk",
            "origin": "web",
            "snippet": "风险证据",
        },
        {
            # 与前一个方向重复的来源只能编号一次。
            "source": "kb/architecture.md",
            "origin": "kb",
            "snippet": "重复证据",
        },
    ]

    def fake_call(user_prompt: str, system_prompt: str, trace_id: str) -> str:
        assert "Critic 返工后的补充资料:" in user_prompt
        assert "### 补充方向：补充 2026 行业风险" in user_prompt
        assert "定向补查后的新证据" in user_prompt
        assert "[3] (web) https://example.com/risk" in user_prompt
        assert user_prompt.count("(kb) kb/architecture.md") == 1
        return "# 使用补查资料的报告"

    monkeypatch.setattr(writer_module, "_call_writer_model", fake_call)

    result = writer_node(state)

    assert result["final_report"] == "# 使用补查资料的报告"


def test_writer_degrades_gracefully_when_research_results_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = make_state()
    state["research_results"] = {}
    state["citations"] = {}

    def fake_call(user_prompt: str, system_prompt: str, trace_id: str) -> str:
        assert user_prompt.count("资料不足或未检索到结果。") == 2
        assert "结构化引用（写作时用 [编号] 对应来源）:\n\n无" in user_prompt
        return "# 降级报告"

    monkeypatch.setattr(writer_module, "_call_writer_model", fake_call)

    result = writer_node(state)

    assert result == {"final_report": "# 降级报告"}


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
