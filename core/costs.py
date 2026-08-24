"""LLM token 成本换算。"""

from __future__ import annotations

import logging

from core.config import get_settings

logger = logging.getLogger(__name__)

TOKENS_PER_MILLION = 1_000_000


def estimate(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    cache_hit_tokens: int = 0,
    cache_miss_tokens: int | None = None,
) -> float:
    """按当前价格配置估算一次 LLM 调用成本，单位由配置指定。"""
    _validate_token_count("prompt_tokens", prompt_tokens)
    _validate_token_count("completion_tokens", completion_tokens)
    _validate_token_count("cache_hit_tokens", cache_hit_tokens)

    if cache_miss_tokens is None:
        cache_miss_tokens = prompt_tokens - cache_hit_tokens
    _validate_token_count("cache_miss_tokens", cache_miss_tokens)

    if cache_hit_tokens + cache_miss_tokens > prompt_tokens:
        raise ValueError("cache token counts must not exceed prompt_tokens")

    settings = get_settings()
    pricing = settings.model_pricing.get(model)
    if pricing is None:
        logger.warning("unknown model pricing model=%s; returning zero cost", model)
        return 0.0

    input_cost = (
        cache_hit_tokens * pricing.input_cache_hit
        + cache_miss_tokens * pricing.input_cache_miss
    ) / TOKENS_PER_MILLION
    output_cost = completion_tokens * pricing.output / TOKENS_PER_MILLION
    return input_cost + output_cost


def _validate_token_count(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


__all__ = ["estimate"]

