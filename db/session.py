"""数据库连接与会话管理（Phase 19）。

**核心设计：落库是可选的。**

``DATABASE_URL`` 为空时，:func:`is_enabled` 返回 ``False``，整个持久化层静默跳过。
这是刻意的 —— v2 的既有优点是"没有任何外部依赖也能跑完整离线测试"，
Phase 19 不能把它弄丢。研究助手丢一条任务记录不是灾难性的，所以这里选 fail-open。

> 对照：合规审查版（金融版）的同一层是 **fail-closed** —— 写不进库就中止整个任务，
> 因为"没留痕的审查结论"在合规场景下比报错更危险。同一段代码，两种场景，
> 失效策略相反。这是有意的，不是疏漏。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from core.config import Settings, get_settings
from db.models import Base

logger = logging.getLogger(__name__)


def is_enabled(settings: Settings | None = None) -> bool:
    """是否配置了数据库。未配置时全链路落库跳过。"""
    current = settings or get_settings()
    return bool(current.database_url.strip())


@lru_cache(maxsize=4)
def _build_engine(url: str, echo: bool) -> Engine:
    """按 URL 缓存 Engine。

    Engine 内部维护连接池，是**重对象**，不该每次请求新建一个 ——
    那样连接数会随请求线性增长，很快打满数据库的 max_connections。
    """
    connect_args: dict[str, object] = {}
    if url.startswith("sqlite"):
        # SQLite 默认禁止跨线程复用连接，而 FastAPI / LangGraph 会在线程池里跑同步代码。
        # 仅测试与本地开发用 SQLite，这里放开该检查。
        connect_args["check_same_thread"] = False
    return create_engine(url, echo=echo, future=True, connect_args=connect_args)


def get_engine(settings: Settings | None = None) -> Engine:
    current = settings or get_settings()
    if not is_enabled(current):
        raise RuntimeError("DATABASE_URL is not configured")
    return _build_engine(current.database_url.strip(), current.db_echo)


def drop_all(settings: Settings | None = None) -> None:
    """删掉所有表。**只给测试用**，用于在复用的外部库上隔离用例。"""
    Base.metadata.drop_all(get_engine(settings))


def create_all(settings: Settings | None = None) -> None:
    """按模型直接建表。

    ⚠️ **只给测试和本地快速起步用。** 生产环境一律走 ``alembic upgrade head`` ——
    ``create_all`` 只能建新表，改不了已有表的结构，用它当迁移方案，
    第一次改字段就会卡住。
    """
    Base.metadata.create_all(get_engine(settings))


@contextmanager
def session_scope(settings: Settings | None = None) -> Iterator[Session]:
    """一个事务边界：正常提交，异常回滚，最后一定关闭。

    用上下文管理器而不是让调用方自己 commit/rollback，是为了保证
    "中途异常时不留半条记录" —— 这是 Phase 19 的验收项之一。
    """
    factory = sessionmaker(bind=get_engine(settings), expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine_cache() -> None:
    """清掉 Engine 缓存。测试里换 ``DATABASE_URL`` 后必须调用。"""
    _build_engine.cache_clear()


__all__ = [
    "create_all",
    "drop_all",
    "get_engine",
    "is_enabled",
    "reset_engine_cache",
    "session_scope",
]
