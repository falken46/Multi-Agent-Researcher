"""所有同步与异步 LLM 调用的唯一入口。"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openai import APIStatusError, APITimeoutError, AsyncOpenAI, OpenAI

from core.config import Settings, get_settings
from core.costs import estimate
from core.trace import emit

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """统一 LLM 调用失败。"""


class LLMConfigurationError(LLMError):
    """LLM 配置缺失或无效。"""


class LLMCallError(LLMError):
    """LLM 请求在重试后仍失败。"""


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0


@dataclass(frozen=True)
class LLMResult:
    content: str
    usage: LLMUsage
    latency_ms: float
    cost: float


Messages = Sequence[Mapping[str, str]]


def chat(
    messages: Messages,
    *,
    node: str,
    trace_id: str,
    json_mode: bool = False,
) -> LLMResult:
    """同步调用 LLM，统一处理重试、计量、成本与 trace。"""
    settings = get_settings()
    client = _create_sync_client(settings)
    request = _build_request(messages, settings, json_mode)
    max_attempts = settings.llm_max_retry + 1

    for attempt in range(1, max_attempts + 1):
        started_at = time.perf_counter()
        try:
            logger.info(
                "llm call input node=%s trace_id=%s model=%s messages=%s attempt=%s",
                node,
                trace_id,
                settings.model_name,
                len(messages),
                attempt,
            )
            response = client.chat.completions.create(**request)
            latency_ms = (time.perf_counter() - started_at) * 1000
            result = _build_result(response, settings, latency_ms)
            _emit_llm_call(
                node=node,
                trace_id=trace_id,
                settings=settings,
                attempt=attempt,
                result=result,
            )
            logger.info(
                "llm call output node=%s trace_id=%s chars=%s tokens=%s latency_ms=%.3f",
                node,
                trace_id,
                len(result.content),
                result.usage.total_tokens,
                result.latency_ms,
            )
            return result
        except Exception as exc:
            latency_ms = (time.perf_counter() - started_at) * 1000
            retryable = _is_retryable(exc)
            _emit_failed_llm_call(
                node=node,
                trace_id=trace_id,
                settings=settings,
                attempt=attempt,
                latency_ms=latency_ms,
                exc=exc,
                retryable=retryable,
            )
            if not retryable or attempt >= max_attempts:
                raise LLMCallError(_failure_message(exc, attempt)) from exc
            time.sleep(_retry_delay(attempt))

    raise LLMCallError("LLM call failed without an attempt")


async def achat(
    messages: Messages,
    *,
    node: str,
    trace_id: str,
    json_mode: bool = False,
) -> LLMResult:
    """异步调用 LLM，行为与 :func:`chat` 保持一致。"""
    settings = get_settings()
    client = _create_async_client(settings)
    request = _build_request(messages, settings, json_mode)
    max_attempts = settings.llm_max_retry + 1

    for attempt in range(1, max_attempts + 1):
        started_at = time.perf_counter()
        try:
            logger.info(
                "llm call input node=%s trace_id=%s model=%s messages=%s attempt=%s",
                node,
                trace_id,
                settings.model_name,
                len(messages),
                attempt,
            )
            response = await client.chat.completions.create(**request)
            latency_ms = (time.perf_counter() - started_at) * 1000
            result = _build_result(response, settings, latency_ms)
            _emit_llm_call(
                node=node,
                trace_id=trace_id,
                settings=settings,
                attempt=attempt,
                result=result,
            )
            logger.info(
                "llm call output node=%s trace_id=%s chars=%s tokens=%s latency_ms=%.3f",
                node,
                trace_id,
                len(result.content),
                result.usage.total_tokens,
                result.latency_ms,
            )
            return result
        except Exception as exc:
            latency_ms = (time.perf_counter() - started_at) * 1000
            retryable = _is_retryable(exc)
            _emit_failed_llm_call(
                node=node,
                trace_id=trace_id,
                settings=settings,
                attempt=attempt,
                latency_ms=latency_ms,
                exc=exc,
                retryable=retryable,
            )
            if not retryable or attempt >= max_attempts:
                raise LLMCallError(_failure_message(exc, attempt)) from exc
            await asyncio.sleep(_retry_delay(attempt))

    raise LLMCallError("LLM call failed without an attempt")


def _create_sync_client(settings: Settings) -> OpenAI:
    try:
        api_key = settings.require_deepseek_api_key()
    except ValueError as exc:
        raise LLMConfigurationError(str(exc)) from exc
    return OpenAI(
        api_key=api_key,
        base_url=settings.deepseek_base_url,
        timeout=settings.llm_timeout,
    )


def _create_async_client(settings: Settings) -> AsyncOpenAI:
    try:
        api_key = settings.require_deepseek_api_key()
    except ValueError as exc:
        raise LLMConfigurationError(str(exc)) from exc
    return AsyncOpenAI(
        api_key=api_key,
        base_url=settings.deepseek_base_url,
        timeout=settings.llm_timeout,
    )


def _build_request(
    messages: Messages,
    settings: Settings,
    json_mode: bool,
) -> dict[str, Any]:
    if not messages:
        raise LLMConfigurationError("messages must not be empty")
    request: dict[str, Any] = {
        "model": settings.model_name,
        "messages": [dict(message) for message in messages],
    }
    if json_mode:
        request["response_format"] = {"type": "json_object"}
    return request


def _build_result(response: Any, settings: Settings, latency_ms: float) -> LLMResult:
    try:
        content = response.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError) as exc:
        raise LLMCallError("LLM response did not contain a message") from exc
    if not content.strip():
        raise LLMCallError("LLM returned empty content")

    usage = _extract_usage(getattr(response, "usage", None))
    cost = estimate(
        settings.model_name,
        usage.prompt_tokens,
        usage.completion_tokens,
        cache_hit_tokens=usage.cache_hit_tokens,
        cache_miss_tokens=usage.cache_miss_tokens,
    )
    return LLMResult(
        content=content.strip(),
        usage=usage,
        latency_ms=latency_ms,
        cost=cost,
    )


def _extract_usage(raw_usage: Any) -> LLMUsage:
    prompt_tokens = _usage_int(raw_usage, "prompt_tokens")
    completion_tokens = _usage_int(raw_usage, "completion_tokens")
    cache_hit_tokens = _usage_int(raw_usage, "prompt_cache_hit_tokens")
    if cache_hit_tokens == 0:
        prompt_details = _usage_value(raw_usage, "prompt_tokens_details")
        cache_hit_tokens = _usage_int(prompt_details, "cached_tokens")
    cache_hit_tokens = min(cache_hit_tokens, prompt_tokens)

    raw_cache_miss_tokens = _usage_value(raw_usage, "prompt_cache_miss_tokens")
    if raw_cache_miss_tokens is None:
        cache_miss_tokens = prompt_tokens - cache_hit_tokens
    else:
        cache_miss_tokens = min(
            _coerce_non_negative_int(raw_cache_miss_tokens),
            prompt_tokens - cache_hit_tokens,
        )

    total_tokens = _usage_int(raw_usage, "total_tokens")
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return LLMUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cache_hit_tokens=cache_hit_tokens,
        cache_miss_tokens=cache_miss_tokens,
    )


def _usage_value(raw_usage: Any, name: str) -> Any:
    if raw_usage is None:
        return None
    if isinstance(raw_usage, Mapping):
        return raw_usage.get(name)
    value = getattr(raw_usage, name, None)
    if value is not None:
        return value
    model_extra = getattr(raw_usage, "model_extra", None)
    if isinstance(model_extra, Mapping):
        return model_extra.get(name)
    return None


def _usage_int(raw_usage: Any, name: str) -> int:
    return _coerce_non_negative_int(_usage_value(raw_usage, name))


def _coerce_non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (APITimeoutError, TimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500
    status_code = getattr(exc, "status_code", None)
    return isinstance(status_code, int) and status_code >= 500


def _retry_delay(attempt: int) -> float:
    return min(float(2 ** (attempt - 1)), 8.0)


def _emit_llm_call(
    *,
    node: str,
    trace_id: str,
    settings: Settings,
    attempt: int,
    result: LLMResult,
) -> None:
    emit(
        {
            "trace_id": trace_id,
            "event": "llm_call",
            "node": node,
            "payload": {
                "success": True,
                "model": settings.model_name,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
                "cache_hit_tokens": result.usage.cache_hit_tokens,
                "cache_miss_tokens": result.usage.cache_miss_tokens,
                "latency_ms": result.latency_ms,
                "cost": result.cost,
                "currency": settings.model_pricing_currency,
                "pricing_version": settings.model_pricing_version,
                "attempt": attempt,
            },
        }
    )


def _emit_failed_llm_call(
    *,
    node: str,
    trace_id: str,
    settings: Settings,
    attempt: int,
    latency_ms: float,
    exc: Exception,
    retryable: bool,
) -> None:
    emit(
        {
            "trace_id": trace_id,
            "event": "llm_call",
            "node": node,
            "payload": {
                "success": False,
                "model": settings.model_name,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": latency_ms,
                "cost": 0.0,
                "currency": settings.model_pricing_currency,
                "pricing_version": settings.model_pricing_version,
                "attempt": attempt,
                "retryable": retryable,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        }
    )


def _failure_message(exc: Exception, attempt: int) -> str:
    return f"LLM call failed after {attempt} attempt(s): {exc}"


__all__ = [
    "LLMCallError",
    "LLMConfigurationError",
    "LLMError",
    "LLMResult",
    "LLMUsage",
    "achat",
    "chat",
]
