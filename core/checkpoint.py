"""LangGraph 异步 SQLite Checkpointer 生命周期管理。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from core.config import get_settings


@asynccontextmanager
async def open_sqlite_checkpointer(
    path: Path | None = None,
) -> AsyncIterator[AsyncSqliteSaver]:
    """在同一事件循环中打开并安全关闭持久化 Checkpointer。"""
    checkpoint_path = Path(path or get_settings().checkpoint_db)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        yield saver


__all__ = ["open_sqlite_checkpointer"]
