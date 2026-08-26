"""Docker / Compose 交付配置的离线契约测试。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_backend_dockerfile_is_multi_stage_and_non_root() -> None:
    dockerfile = _read("Dockerfile.backend")

    assert dockerfile.count("FROM ") >= 2
    assert "uv sync --frozen --no-dev --no-install-project" in dockerfile
    assert "COPY --from=builder /app/.venv /app/.venv" in dockerfile
    assert "USER app" in dockerfile
    assert "backend.api:app" in dockerfile
    assert "HEALTHCHECK" in dockerfile


def test_frontend_dockerfile_is_multi_stage_and_non_root() -> None:
    dockerfile = _read("Dockerfile.frontend")

    assert dockerfile.count("FROM ") >= 2
    assert "uv sync --frozen --no-dev --no-install-project" in dockerfile
    assert "COPY --from=builder /app/.venv /app/.venv" in dockerfile
    assert "USER app" in dockerfile
    assert "streamlit" in dockerfile
    assert "_stcore/health" in dockerfile


def test_compose_orders_index_backend_and_frontend() -> None:
    compose = _read("docker-compose.yml")

    assert "indexer:" in compose
    assert 'command: ["python", "-m", "rag.index_cli"' in compose
    assert "condition: service_completed_successfully" in compose
    assert "condition: service_healthy" in compose
    assert "BACKEND_URL: http://backend:8000" in compose


def test_compose_limits_secrets_to_services_that_need_them() -> None:
    compose = _read("docker-compose.yml")
    services = compose.split("services:", maxsplit=1)[1]
    indexer_block, backend_and_frontend = services.split("  backend:", maxsplit=1)
    backend_block, frontend_block = backend_and_frontend.split(
        "  frontend:", maxsplit=1
    )

    assert "env_file:" not in indexer_block
    assert "EMBEDDING_API_KEY:" in indexer_block
    assert "checkpoint-data:/app/data/checkpoints" not in indexer_block
    assert "trace-data:/app/traces" not in indexer_block
    assert "env_file:" in backend_block
    assert "env_file:" not in frontend_block


def test_compose_persists_runtime_data_outside_images() -> None:
    compose = _read("docker-compose.yml")

    for volume_name in (
        "chroma-data",
        "bm25-data",
        "checkpoint-data",
        "trace-data",
        "model-cache",
    ):
        assert f"{volume_name}:" in compose

    assert "CHROMA_DIR: /app/data/chroma" in compose
    assert "BM25_INDEX_PATH: /app/data/bm25/index.pkl" in compose
    assert "CHECKPOINT_DB: /app/data/checkpoints/checkpoints.sqlite" in compose
    assert "TRACE_DIR: /app/traces" in compose


def test_docker_context_excludes_secrets_and_runtime_data() -> None:
    dockerignore = _read(".dockerignore")

    assert ".env\n" in dockerignore
    assert "data/chroma\n" in dockerignore
    assert "data/bm25\n" in dockerignore
    assert "data/checkpoints\n" in dockerignore
    assert "traces\n" in dockerignore


def test_ci_uses_locked_dependencies_and_offline_quality_gates() -> None:
    workflow = _read(".github/workflows/ci.yml")

    assert "permissions:\n  contents: read" in workflow
    assert "uv sync --frozen --group dev" in workflow
    assert "uv run --frozen ruff check ." in workflow
    assert 'uv run --frozen python -m pytest -m "not live"' in workflow
    assert "secrets." not in workflow


def test_ci_actions_are_pinned_to_full_commit_shas() -> None:
    workflow = _read(".github/workflows/ci.yml")

    assert (
        "actions/checkout@11d5960a326750d5838078e36cf38b85af677262" in workflow
    )
    assert (
        "astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9" in workflow
    )
