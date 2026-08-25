from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest

from agents.graph import create_initial_state
from agents.state import ResearchState

critic_module = importlib.import_module("agents.critic")


def make_state() -> ResearchState:
    state = create_initial_state("AI Agent 趋势", trace_id="trace-critic")
    state["sub_questions"] = ["架构趋势是什么？"]
    state["research_results"] = {"架构趋势是什么？": "研究资料"}
    state["citations"] = {
        "架构趋势是什么？": [
            {
                "source": "kb/architecture.md",
                "origin": "kb",
                "snippet": "架构资料",
            }
        ]
    }
    return state


def llm_result(content: str) -> SimpleNamespace:
    return SimpleNamespace(content=content)


@pytest.mark.asyncio
async def test_critic_parses_structured_result_and_normalizes_missing_aspects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_achat(messages, *, node: str, trace_id: str, json_mode: bool):
        assert node == "critic"
        assert trace_id == "trace-critic"
        assert json_mode is True
        request = json.loads(messages[1]["content"])
        assert request["topic"] == "AI Agent 趋势"
        assert request["citations"]["架构趋势是什么？"][0]["origin"] == "kb"
        return llm_result(
            json.dumps(
                {
                    "quality_score": 0.82,
                    "critique": "覆盖较完整，但还缺少风险数据。",
                    "missing_aspects": [" 风险数据 ", "风险数据", "", 42],
                },
                ensure_ascii=False,
            )
        )

    monkeypatch.setattr(critic_module, "achat", fake_achat)

    result = await critic_module.critic_node(make_state())

    assert result == {
        "quality_score": pytest.approx(0.82),
        "quality_history": [pytest.approx(0.82)],
        "critique": "覆盖较完整，但还缺少风险数据。",
        "missing_aspects": ["风险数据"],
    }


@pytest.mark.asyncio
async def test_critic_accepts_json_code_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_achat(messages, *, node: str, trace_id: str, json_mode: bool):
        return llm_result(
            """```json
            {
              "quality_score": 0.75,
              "critique": "达到写作标准。",
              "missing_aspects": []
            }
            ```"""
        )

    monkeypatch.setattr(critic_module, "achat", fake_achat)

    result = await critic_module.critic_node(make_state())

    assert result["quality_score"] == pytest.approx(0.75)
    assert result["critique"] == "达到写作标准。"
    assert result["missing_aspects"] == []
    assert "errors" not in result


@pytest.mark.asyncio
async def test_critic_degrades_invalid_json_to_neutral_score_without_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_achat(messages, *, node: str, trace_id: str, json_mode: bool):
        return llm_result("not json")

    state = make_state()
    state["errors"] = ["Researcher: previous warning"]
    state["quality_history"] = [0.6]
    monkeypatch.setattr(critic_module, "achat", fake_achat)

    result = await critic_module.critic_node(state)

    assert result["quality_score"] == pytest.approx(0.5)
    assert result["quality_history"] == [0.6, 0.5]
    assert result["missing_aspects"] == []
    assert result["errors"][0] == "Researcher: previous warning"
    assert result["errors"][1].startswith("Critic: structured output is not valid JSON")
    assert "中性分数降级" in result["critique"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_score", "expected_score"),
    [(-2, 0.0), (1.8, 1.0)],
)
async def test_critic_clamps_out_of_range_score(
    monkeypatch: pytest.MonkeyPatch,
    raw_score: float,
    expected_score: float,
) -> None:
    async def fake_achat(messages, *, node: str, trace_id: str, json_mode: bool):
        return llm_result(
            json.dumps(
                {
                    "quality_score": raw_score,
                    "critique": "分数边界测试。",
                    "missing_aspects": [],
                }
            )
        )

    monkeypatch.setattr(critic_module, "achat", fake_achat)

    result = await critic_module.critic_node(make_state())

    assert result["quality_score"] == expected_score
    assert result["quality_history"] == [expected_score]


@pytest.mark.asyncio
async def test_critic_rejects_nan_score_and_degrades_neutrally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_achat(messages, *, node: str, trace_id: str, json_mode: bool):
        return llm_result(
            '{"quality_score": NaN, "critique": "非法分数", "missing_aspects": ["X"]}'
        )

    monkeypatch.setattr(critic_module, "achat", fake_achat)

    result = await critic_module.critic_node(make_state())

    assert result["quality_score"] == pytest.approx(0.5)
    assert result["quality_history"] == [0.5]
    assert result["missing_aspects"] == []
    assert result["errors"][0] == "Critic: quality_score must be a finite number"


@pytest.mark.asyncio
async def test_critic_degrades_empty_research_results_without_calling_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unexpected_achat(*args, **kwargs):
        raise AssertionError("empty input must fail before LLM call")

    state = make_state()
    state["research_results"] = {}
    monkeypatch.setattr(critic_module, "achat", unexpected_achat)

    result = await critic_module.critic_node(state)

    assert result["quality_score"] == 0.5
    assert result["missing_aspects"] == []
    assert "research_results must not be empty" in result["errors"][0]
