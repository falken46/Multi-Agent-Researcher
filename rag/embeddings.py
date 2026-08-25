"""可插拔的本地、远程与测试 embedding 后端。"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

import requests

from core.config import Settings, get_settings

Embedding = list[float]


class EmbeddingBackend(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]: ...

    def embed_query(self, text: str) -> Embedding: ...


class FakeEmbeddingBackend:
    """基于词项哈希的确定性向量，仅用于测试与离线验收。"""

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        self.dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> Embedding:
        return self._embed(text)

    def _embed(self, text: str) -> Embedding:
        vector = [0.0] * self.dimension
        for token in _semantic_tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class FastEmbedBackend:
    """FastEmbed ONNX 文本向量实现，模型在首次实例化时加载。"""

    def __init__(self, model_name: str) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError("fastembed backend requires the fastembed package") from exc
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [vector.tolist() for vector in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> Embedding:
        vectors = list(self._model.query_embed([text]))
        if not vectors:
            raise RuntimeError("fastembed returned no query embedding")
        return vectors[0].tolist()


class RemoteEmbeddingBackend:
    """调用 OpenAI 兼容的 HTTP embedding 接口。"""

    def __init__(
        self,
        *,
        url: str,
        model_name: str,
        api_key: str = "",
        timeout: float = 30.0,
    ) -> None:
        if not url.strip():
            raise ValueError("EMBEDDING_REMOTE_URL is required for remote backend")
        self._url = url.strip()
        self._model_name = model_name
        self._api_key = api_key.strip()
        self._timeout = timeout

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        if not texts:
            return []
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        response = requests.post(
            self._url,
            headers=headers,
            json={"model": self._model_name, "input": list(texts)},
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, list):
            raise RuntimeError("remote embedding response missing data list")
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        embeddings = [item.get("embedding") for item in ordered]
        if len(embeddings) != len(texts) or not all(
            isinstance(item, list) for item in embeddings
        ):
            raise RuntimeError("remote embedding response has invalid vector count")
        return [[float(value) for value in item] for item in embeddings]

    def embed_query(self, text: str) -> Embedding:
        vectors = self.embed_documents([text])
        return vectors[0]


def create_embedding_backend(settings: Settings | None = None) -> EmbeddingBackend:
    """根据集中配置创建 embedding 后端。"""
    current = settings or get_settings()
    if current.embedding_backend == "fake":
        return FakeEmbeddingBackend()
    if current.embedding_backend == "fastembed":
        return FastEmbedBackend(current.embedding_model)
    return RemoteEmbeddingBackend(
        url=current.embedding_remote_url,
        model_name=current.embedding_model,
        api_key=current.embedding_api_key.get_secret_value(),
        timeout=current.embedding_timeout,
    )


def _semantic_tokens(text: str) -> list[str]:
    normalized = text.lower()
    tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", normalized)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", normalized)
    for run in chinese_runs:
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


__all__ = [
    "Embedding",
    "EmbeddingBackend",
    "FakeEmbeddingBackend",
    "FastEmbedBackend",
    "RemoteEmbeddingBackend",
    "create_embedding_backend",
]
