"""落库的安全入口（Phase 19）。

:mod:`db.repository` 是纯粹的数据访问层：它要求调用方传入 ``Session``，
异常照常抛出，方便单元测试注入和断言。

本模块在它外面包一层，负责三件事：

1. 检查 ``DATABASE_URL`` 有没有配置 —— 没配就直接跳过
2. 开事务边界
3. **吞掉异常，只记日志**

第 3 条是刻意的 fail-open：研究助手丢一条任务记录不是灾难，
但因为落库失败而让用户的整份研究报告返回不了，是不可接受的。

> ⚠️ 对照：合规审查版（金融版）的同一层是 **fail-closed** —— 写不进库就中止任务。
> 因为"一份没留痕的合规审查结论"比报错危险得多。同一段代码、两种场景、相反策略。
> 见 `实习材料/证券合规审查Agent/ARCHITECTURE.md`。
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from core.config import Settings, get_settings
from db import repository
from db.session import is_enabled, session_scope

logger = logging.getLogger(__name__)


def persist_research_run(
    *,
    thread_id: str,
    state: Mapping[str, Any],
    status: str = "completed",
    actor: str | None = None,
    settings: Settings | None = None,
) -> int | None:
    """把一次研究任务落库。返回任务主键；未启用或失败时返回 ``None``。

    永远不抛异常 —— 调用方不需要 try/except 包裹。
    """
    current = settings or get_settings()
    if not is_enabled(current):
        return None

    try:
        with session_scope(current) as session:
            task = repository.save_research_run(
                session,
                thread_id=thread_id,
                state=state,
                status=status,
                actor=actor,
                model_name=current.model_name,
                vector_backend=current.vector_backend,
            )
            task_id = task.id
        logger.info(
            "persisted research run thread_id=%s task_id=%s status=%s",
            thread_id,
            task_id,
            status,
        )
        return task_id
    except Exception:
        # 落库失败不能影响主流程返回结果，但必须留下完整栈供排查
        logger.exception("failed to persist research run thread_id=%s", thread_id)
        return None


__all__ = ["persist_research_run"]
