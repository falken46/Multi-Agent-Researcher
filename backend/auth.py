"""API Key 鉴权与调用主体解析（Phase 20）。

v2 的接口是完全裸奔的：trace 里记了模型、token、耗时、成本，**唯独没有"是谁在调"**。
可观测缺了调用主体这一维，事后就回答不了「这次任务是谁发起的」。

设计上刻意保持简单，并且**默认关闭**：

- ``API_KEYS`` 留空 → 不鉴权，所有请求放行，``actor`` 记为 ``anonymous``
- 配了 ``API_KEYS`` → 请求必须带 ``X-API-Key``，否则 401

默认关闭是为了保住"clone 下来直接能跑"，以及不破坏既有的前端与测试。

⚠️ **这不是生产级方案。** 明文 key 写在配置里、没有轮换、没有过期、没有权限分级。
它解决的是"把调用主体接进可观测链路"，不是"保护一个对外服务"。
真要对外暴露应该换成 OAuth / JWT + 密钥管理服务。
"""

from __future__ import annotations

import logging
import secrets

from fastapi import Header, HTTPException

from core.config import Settings, get_settings

logger = logging.getLogger(__name__)

ANONYMOUS = "anonymous"


def parse_api_keys(raw: str) -> dict[str, str]:
    """解析 ``key1:研究所,key2:合规部`` 形式的配置，返回 ``key -> 主体名``。

    没写主体名时（只有 ``key1``）主体名回退成 key 本身的前 8 位，
    避免把完整密钥当成标识写进日志和数据库。
    """
    mapping: dict[str, str] = {}
    for item in raw.split(","):
        entry = item.strip()
        if not entry:
            continue
        key, _, actor = entry.partition(":")
        key = key.strip()
        if not key:
            continue
        mapping[key] = actor.strip() or f"key-{key[:8]}"
    return mapping


def resolve_actor(
    api_key: str | None,
    *,
    settings: Settings | None = None,
) -> str:
    """校验 key 并返回调用主体名；未配置 ``API_KEYS`` 时直接放行。"""
    current = settings or get_settings()
    configured = parse_api_keys(current.api_keys)

    if not configured:
        return ANONYMOUS

    if not api_key:
        raise HTTPException(status_code=401, detail="missing X-API-Key")

    # 用常量时间比较，避免通过响应时间差逐字节猜测 key。
    # 这个项目里意义有限（key 是明文配置的），但比较本身零成本，没有理由写错。
    for candidate, actor in configured.items():
        if secrets.compare_digest(api_key, candidate):
            return actor

    logger.warning("rejected request with unknown api key prefix=%s", api_key[:6])
    raise HTTPException(status_code=401, detail="invalid X-API-Key")


async def require_actor(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    """FastAPI 依赖：解析并返回调用主体。"""
    return resolve_actor(x_api_key)


__all__ = ["ANONYMOUS", "parse_api_keys", "require_actor", "resolve_actor"]
