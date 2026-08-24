from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from core.config import clear_settings_cache
from core.trace import new_trace_id


@pytest.fixture(autouse=True)
def reset_settings_cache(monkeypatch: pytest.MonkeyPatch):
    """隔离测试间的环境配置，并避免单元测试写入项目 trace 目录。"""
    monkeypatch.setenv("TRACE_ENABLED", "false")
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture
def runtime_dir() -> Path:
    """在工作区内提供可清理的运行时目录，绕开沙箱临时目录 ACL。"""
    runtime_root = Path(".test-runtime")
    path = runtime_root / new_trace_id()
    yield path
    shutil.rmtree(path, ignore_errors=True)
    try:
        runtime_root.rmdir()
    except OSError:
        pass
