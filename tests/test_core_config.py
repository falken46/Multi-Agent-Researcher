from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.config import Settings, clear_settings_cache, get_settings


def test_settings_defaults_are_typed() -> None:
    settings = Settings(_env_file=None)

    assert settings.model_name == "deepseek-v4-flash"
    assert settings.llm_timeout == 60.0
    assert settings.max_retry == 2
    assert settings.embedding_backend == "fastembed"
    assert settings.chroma_collection == "deepresearch_kb"
    assert settings.vector_search_enabled is True
    assert settings.bm25_search_enabled is True
    assert Settings.model_fields["trace_enabled"].default is True
    assert settings.model_pricing["deepseek-v4-flash"].output == 2.0


def test_settings_load_and_validate_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_TIMEOUT", "12.5")
    monkeypatch.setenv("MAX_RETRY", "4")
    monkeypatch.setenv("TRACE_ENABLED", "false")
    clear_settings_cache()

    settings = get_settings()

    assert settings.llm_timeout == 12.5
    assert settings.max_retry == 4
    assert settings.trace_enabled is False


def test_settings_reject_invalid_chunk_configuration() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)


def test_required_api_key_errors_are_explicit() -> None:
    settings = Settings(_env_file=None, deepseek_api_key="")

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY is not configured"):
        settings.require_deepseek_api_key()


def test_required_api_key_must_be_ascii() -> None:
    settings = Settings(_env_file=None, deepseek_api_key="sk-测试")

    with pytest.raises(ValueError, match="ASCII"):
        settings.require_deepseek_api_key()
