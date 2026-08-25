from __future__ import annotations

from pathlib import Path

import pytest

from agents.graph import build_graph, create_initial_state
from agents.state import ResearchState
from core.checkpoint import open_sqlite_checkpointer


@pytest.mark.asyncio
async def test_sqlite_checkpoint_resumes_after_saver_is_closed_and_reopened(
    runtime_dir: Path,
) -> None:
    checkpoint_path = runtime_dir / "checkpoints.sqlite"
    config = {"configurable": {"thread_id": "resume-after-reopen"}}
    calls: list[str] = []

    def fake_planner(state: ResearchState) -> dict[str, object]:
        calls.append("planner")
        return {"sub_questions": ["Q1"]}

    async def fake_researcher(state: ResearchState) -> dict[str, object]:
        calls.append("researcher")
        return {
            "research_results": {"Q1": "R1"},
            "citations": {
                "Q1": [
                    {
                        "source": "local.md",
                        "origin": "kb",
                        "snippet": "evidence",
                    }
                ]
            },
            "errors": [],
        }

    async def fake_critic(state: ResearchState) -> dict[str, object]:
        calls.append("critic")
        assert state["research_results"] == {"Q1": "R1"}
        return {
            "critique": "sufficient",
            "quality_score": 0.9,
            "quality_history": [0.9],
            "missing_aspects": [],
        }

    def fake_writer(state: ResearchState) -> dict[str, object]:
        calls.append("writer")
        assert state["quality_score"] == 0.9
        return {"final_report": "# resumed report"}

    initial_state = create_initial_state(
        "checkpoint topic",
        trace_id="checkpoint-trace",
    )

    async with open_sqlite_checkpointer(checkpoint_path) as saver:
        interrupted_graph = build_graph(
            planner=fake_planner,
            researcher=fake_researcher,
            critic=fake_critic,
            writer=fake_writer,
            checkpointer=saver,
            interrupt_after=["researcher"],
        )
        interrupted_state = await interrupted_graph.ainvoke(
            initial_state,
            config,
            durability="sync",
        )
        snapshot = await interrupted_graph.aget_state(config)

    assert checkpoint_path.is_file()
    assert calls == ["planner", "researcher"]
    assert interrupted_state["research_results"] == {"Q1": "R1"}
    assert snapshot.next == ("critic",)

    async with open_sqlite_checkpointer(checkpoint_path) as saver:
        resumed_graph = build_graph(
            planner=fake_planner,
            researcher=fake_researcher,
            critic=fake_critic,
            writer=fake_writer,
            checkpointer=saver,
            interrupt_after=["researcher"],
        )
        final_state = await resumed_graph.ainvoke(
            None,
            config,
            durability="sync",
        )
        final_snapshot = await resumed_graph.aget_state(config)

    assert calls == ["planner", "researcher", "critic", "writer"]
    assert final_state["trace_id"] == "checkpoint-trace"
    assert final_state["final_report"] == "# resumed report"
    assert final_snapshot.next == ()
