from __future__ import annotations

import logging

import pytest

from core.costs import estimate


def test_estimate_uses_configured_model_prices() -> None:
    cost = estimate(
        "deepseek-v4-flash",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )

    assert cost == pytest.approx(3.0)


def test_estimate_distinguishes_cache_hit_and_miss_tokens() -> None:
    cost = estimate(
        "deepseek-v4-flash",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        cache_hit_tokens=500_000,
        cache_miss_tokens=500_000,
    )

    assert cost == pytest.approx(2.51)


def test_estimate_unknown_model_returns_zero(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        cost = estimate("unknown-model", 10, 20)

    assert cost == 0.0
    assert "unknown model pricing" in caplog.text


def test_estimate_rejects_invalid_token_counts() -> None:
    with pytest.raises(ValueError, match="prompt_tokens"):
        estimate("deepseek-v4-flash", -1, 1)

    with pytest.raises(ValueError, match="must not exceed"):
        estimate(
            "deepseek-v4-flash",
            10,
            1,
            cache_hit_tokens=6,
            cache_miss_tokens=5,
        )

