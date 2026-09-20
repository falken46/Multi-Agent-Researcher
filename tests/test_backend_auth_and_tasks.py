"""接口鉴权与任务查询接口测试（Phase 20）。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.api import create_app
from backend.auth import ANONYMOUS, parse_api_keys, resolve_actor
from backend.streaming import stream_research_progress
from core.config import Settings, clear_settings_cache
from db import repository
from db.models import ResearchTask
from db.session import create_all, reset_engine_cache, session_scope

# ---------- API Key 解析与校验 ----------


def test_parse_api_keys_supports_actor_names() -> None:
    mapping = parse_api_keys("k1:研究所, k2:合规部 ,")
    assert mapping == {"k1": "研究所", "k2": "合规部"}


def test_parse_api_keys_falls_back_to_key_prefix() -> None:
    """没写主体名时用 key 前 8 位，避免把完整密钥当标识写进日志和数据库。"""
    mapping = parse_api_keys("abcdefghijklmnop")
    assert mapping == {"abcdefghijklmnop": "key-abcdefgh"}


def test_auth_disabled_by_default() -> None:
    """未配置 API_KEYS 时全部放行 —— 保住 clone 下来直接能跑。"""
    settings = Settings(_env_file=None, api_keys="")
    assert resolve_actor(None, settings=settings) == ANONYMOUS
    assert resolve_actor("whatever", settings=settings) == ANONYMOUS


def test_auth_rejects_missing_and_wrong_key() -> None:
    settings = Settings(_env_file=None, api_keys="good:研究所")
    with pytest.raises(HTTPException) as missing:
        resolve_actor(None, settings=settings)
    assert missing.value.status_code == 401

    with pytest.raises(HTTPException) as wrong:
        resolve_actor("bad", settings=settings)
    assert wrong.value.status_code == 401


def test_auth_resolves_actor_name() -> None:
    settings = Settings(_env_file=None, api_keys="good:研究所,other:合规部")
    assert resolve_actor("good", settings=settings) == "研究所"
    assert resolve_actor("other", settings=settings) == "合规部"


# ---------- 接口层 ----------


@pytest.fixture
def client_with_db(runtime_dir: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    db_path = (runtime_dir / "api.db").as_posix()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("API_KEYS", "secret:研究所")
    clear_settings_cache()
    reset_engine_cache()
    create_all()
    yield TestClient(create_app())
    reset_engine_cache()
    clear_settings_cache()


def _seed(thread_id: str = "t-api") -> None:
    state = {
        "topic": "RRF 为什么有效",
        "trace_id": "trace-api",
        "sub_questions": ["RRF 的公式", "为什么不用加权求和"],
        "final_report": "报告正文",
        "quality_score": 0.9,
        "revision_count": 0,
        "fallback_queries": ["为什么不用加权求和"],
        "citations": {
            "RRF 的公式": [
                {"source": "13_rrf.md", "origin": "kb", "snippet": "定义", "score": 0.8},
                {"source": "36_fusion.md", "origin": "kb", "snippet": "融合"},
            ]
        },
        "usage": {"total_tokens": 100, "llm_calls": 3},
    }
    with session_scope() as session:
        repository.save_research_run(
            session, thread_id=thread_id, state=state, actor="研究所"
        )


def test_health_needs_no_key(client_with_db: TestClient) -> None:
    """健康检查不鉴权 —— 否则容器 healthcheck 也要配密钥。"""
    assert client_with_db.get("/health").status_code == 200


def test_task_endpoints_require_api_key(client_with_db: TestClient) -> None:
    assert client_with_db.get("/tasks").status_code == 401
    assert client_with_db.get("/tasks/t-api").status_code == 401


def test_list_tasks_returns_persisted_runs(client_with_db: TestClient) -> None:
    _seed()
    response = client_with_db.get("/tasks", headers={"X-API-Key": "secret"})
    assert response.status_code == 200
    body = response.json()
    assert body["actor"] == "研究所"
    assert body["count"] == 1
    assert body["tasks"][0]["thread_id"] == "t-api"
    assert body["tasks"][0]["usage"]["total_tokens"] == 100


def test_get_task_includes_sub_questions(client_with_db: TestClient) -> None:
    _seed()
    response = client_with_db.get("/tasks/t-api", headers={"X-API-Key": "secret"})
    assert response.status_code == 200
    body = response.json()
    assert [q["question"] for q in body["sub_questions"]] == [
        "RRF 的公式",
        "为什么不用加权求和",
    ]
    assert body["sub_questions"][1]["fallback_triggered"] is True


def test_get_report_returns_citations_with_rank(client_with_db: TestClient) -> None:
    """引用必须带名次；分数取不到时是 null 而不是 0。"""
    _seed()
    response = client_with_db.get(
        "/tasks/t-api/report", headers={"X-API-Key": "secret"}
    )
    assert response.status_code == 200
    citations = response.json()["citations"]
    assert [c["rank"] for c in citations] == [0, 1]
    assert citations[0]["retrieval_score"] == pytest.approx(0.8)
    assert citations[1]["retrieval_score"] is None


def test_unknown_task_returns_404(client_with_db: TestClient) -> None:
    assert (
        client_with_db.get("/tasks/nope", headers={"X-API-Key": "secret"}).status_code
        == 404
    )


def test_task_endpoints_return_503_without_database(
    runtime_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """没开落库时返回 503，而不是空列表。

    空列表会被读成"确实没有任务"，但真相是"这个部署根本没开落库"——
    两者含义完全不同，不能混。
    """
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("API_KEYS", "")
    clear_settings_cache()
    reset_engine_cache()
    client = TestClient(create_app())
    assert client.get("/tasks").status_code == 503
    clear_settings_cache()


# ---------- SSE 执行与任务生命周期落库 ----------


class _SuccessfulLifecycleGraph:
    checkpointer = None

    async def astream(
        self,
        graph_input: Any,
        config: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        yield ("updates", {"planner": {"sub_questions": ["RRF 是什么"]}})
        yield (
            "updates",
            {
                "writer": {
                    "final_report": "# 生命周期测试报告",
                    "quality_score": 0.88,
                    "citations": {
                        "RRF 是什么": [
                            {
                                "source": "13_rrf.md",
                                "origin": "kb",
                                "snippet": "RRF 定义",
                                "score": 0.81,
                            }
                        ]
                    },
                }
            },
        )


class _FailingLifecycleGraph:
    checkpointer = None

    async def astream(
        self,
        graph_input: Any,
        config: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        if False:  # pragma: no cover - 让本函数保持 async iterator 形态
            yield None
        raise RuntimeError("graph boom")


class _ResumableLifecycleGraph(_SuccessfulLifecycleGraph):
    async def aget_state(self, config: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(
            values={
                "topic": "恢复任务",
                "trace_id": "lifecycle-resume",
                "sub_questions": ["RRF 是什么"],
                "errors": ["previous failure"],
            }
        )


@pytest.mark.asyncio
async def test_stream_persists_running_then_completed(
    client_with_db: TestClient,
) -> None:
    stream = stream_research_progress(
        "任务生命周期",
        _SuccessfulLifecycleGraph(),
        thread_id="lifecycle-success",
        actor="研究所",
    )

    start_event = await anext(stream)
    assert start_event["event"] == "start"
    with session_scope() as session:
        running = repository.get_task_by_thread_id(session, "lifecycle-success")
        assert running is not None
        assert running.status == "running"
        assert running.finished_at is None
        assert running.actor == "研究所"

    remaining = [event async for event in stream]
    assert remaining[-1]["event"] == "complete"
    with session_scope() as session:
        completed = repository.get_task_by_thread_id(session, "lifecycle-success")
        assert completed is not None
        assert completed.status == "completed"
        assert completed.finished_at is not None
        assert len(completed.reports) == 1
        assert completed.reports[0].content == "# 生命周期测试报告"


@pytest.mark.asyncio
async def test_stream_persists_failed_status_on_graph_error(
    client_with_db: TestClient,
) -> None:
    events = [
        event
        async for event in stream_research_progress(
            "失败任务",
            _FailingLifecycleGraph(),
            thread_id="lifecycle-failed",
            actor="研究所",
        )
    ]

    assert [event["event"] for event in events] == ["start", "error"]
    with session_scope() as session:
        failed = repository.get_task_by_thread_id(session, "lifecycle-failed")
        assert failed is not None
        assert failed.status == "failed"
        assert failed.finished_at is not None
        assert failed.actor == "研究所"
        assert failed.reports == []


@pytest.mark.asyncio
async def test_resume_updates_existing_task_instead_of_inserting_duplicate(
    client_with_db: TestClient,
) -> None:
    _ = [
        event
        async for event in stream_research_progress(
            "恢复任务",
            _FailingLifecycleGraph(),
            thread_id="lifecycle-resume",
            actor="研究所",
        )
    ]

    resumed = stream_research_progress(
        "恢复任务",
        _ResumableLifecycleGraph(),
        thread_id="lifecycle-resume",
        resume=True,
        actor="合规部",
    )
    start_event = await anext(resumed)
    assert start_event["event"] == "start"
    with session_scope() as session:
        running = repository.get_task_by_thread_id(session, "lifecycle-resume")
        assert running is not None
        assert running.status == "running"
        assert running.finished_at is None

    _ = [event async for event in resumed]
    with session_scope() as session:
        rows = session.scalars(
            select(ResearchTask).where(
                ResearchTask.thread_id == "lifecycle-resume"
            )
        ).all()
        assert len(rows) == 1
        assert rows[0].status == "completed"
        assert rows[0].actor == "合规部"
        assert len(rows[0].reports) == 1
