"""项目配置的唯一声明与读取入口。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelPricing(BaseModel):
    """单个模型按百万 token 计价的人民币价格。"""

    input_cache_hit: float = Field(ge=0)
    input_cache_miss: float = Field(ge=0)
    output: float = Field(ge=0)


def _default_model_pricing() -> dict[str, ModelPricing]:
    return {
        "deepseek-v4-flash": ModelPricing(
            input_cache_hit=0.02,
            input_cache_miss=1.0,
            output=2.0,
        ),
        "deepseek-v4-pro": ModelPricing(
            input_cache_hit=0.025,
            input_cache_miss=3.0,
            output=6.0,
        ),
    }


class Settings(BaseSettings):
    """从环境变量与项目根目录的 ``.env`` 加载全部配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    deepseek_api_key: SecretStr = SecretStr("")
    deepseek_base_url: str = "https://api.deepseek.com"
    model_name: str = "deepseek-v4-flash"
    llm_timeout: float = Field(default=60.0, gt=0)
    llm_max_retry: int = Field(default=3, ge=0)
    model_pricing: dict[str, ModelPricing] = Field(
        default_factory=_default_model_pricing
    )
    model_pricing_currency: str = "CNY"
    model_pricing_version: str = "2026-08-24"

    # 联网搜索
    search_provider: Literal["tavily", "duckduckgo"] = "tavily"
    tavily_api_key: SecretStr = SecretStr("")

    # 检索层（Phase 11 使用，Phase 10 统一声明）
    embedding_backend: Literal["fastembed", "remote", "fake"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_remote_url: str = ""
    embedding_api_key: SecretStr = SecretStr("")
    embedding_timeout: float = Field(default=30.0, gt=0)
    chroma_dir: Path = Path("data/chroma")
    chroma_collection: str = "deepresearch_kb"
    bm25_index_path: Path = Path("data/bm25/index.pkl")
    kb_dir: Path = Path("data/kb")
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=80, ge=0)
    retrieval_top_k: int = Field(default=20, gt=0)
    vector_search_enabled: bool = True
    bm25_search_enabled: bool = True
    rerank_backend: Literal["onnx", "llm", "none"] = "onnx"
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_top_n: int = Field(default=5, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    kb_score_threshold: float = Field(default=0.35, ge=0, le=1)

    # 编排层
    research_concurrency: int = Field(default=3, gt=0)
    max_retry: int = Field(default=2, ge=0)
    max_revision: int = Field(default=2, ge=0)
    quality_threshold: float = Field(default=0.7, ge=0, le=1)
    checkpoint_db: Path = Path("data/checkpoints.sqlite")

    # 可观测
    trace_dir: Path = Path("traces")
    trace_enabled: bool = True

    # 前端
    backend_url: str = "http://127.0.0.1:8000"

    @model_validator(mode="after")
    def validate_cross_field_constraints(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if not self.deepseek_base_url.strip():
            raise ValueError("DEEPSEEK_BASE_URL must not be empty")
        if not self.model_name.strip():
            raise ValueError("MODEL_NAME must not be empty")
        if not self.chroma_collection.strip():
            raise ValueError("CHROMA_COLLECTION must not be empty")
        if not self.embedding_model.strip():
            raise ValueError("EMBEDDING_MODEL must not be empty")
        if not self.rerank_model.strip():
            raise ValueError("RERANK_MODEL must not be empty")
        return self

    def require_deepseek_api_key(self) -> str:
        """返回可用的 DeepSeek Key；缺失或含非 ASCII 字符时明确失败。"""
        return _require_ascii_secret(
            "DEEPSEEK_API_KEY",
            self.deepseek_api_key.get_secret_value(),
        )

    def require_tavily_api_key(self) -> str:
        """返回可用的 Tavily Key；缺失或含非 ASCII 字符时明确失败。"""
        return _require_ascii_secret(
            "TAVILY_API_KEY",
            self.tavily_api_key.get_secret_value(),
        )


def _require_ascii_secret(name: str, value: str) -> str:
    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"{name} is not configured")
    try:
        normalized_value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must contain ASCII characters only") from exc
    return normalized_value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程内缓存的项目配置。"""
    return Settings()


def clear_settings_cache() -> None:
    """清空配置缓存，供测试和显式重载配置使用。"""
    get_settings.cache_clear()


__all__ = [
    "ModelPricing",
    "Settings",
    "clear_settings_cache",
    "get_settings",
]
