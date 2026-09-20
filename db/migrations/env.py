"""Alembic 迁移环境。

与默认模板的两处差别：

1. **连接串从项目配置读，不从 ``alembic.ini`` 读。** 这样 `.env` / 环境变量是唯一真相源，
   不会出现「应用连 A 库、迁移改 B 库」这种最难查的事故。
2. ``target_metadata`` 指向 :class:`db.models.Base`，``--autogenerate`` 才能对比出差异。
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# 让 alembic 能 import 到项目包（alembic 是从项目根目录之外调起的）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.config import get_settings  # noqa: E402
from db.models import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    url = get_settings().database_url.strip()
    if not url:
        raise RuntimeError(
            "DATABASE_URL 未配置。迁移必须显式指定目标库，"
            "不提供默认值是为了避免误改到错误的数据库。"
        )
    return url


def run_migrations_offline() -> None:
    """离线模式：只生成 SQL 文本，不真正连库。"""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # SQLite 不支持大多数 ALTER，必须用「建新表 → 拷数据 → 换名」的批处理模式。
        # 本项目测试跑 SQLite、生产跑 PostgreSQL，开着它两边都安全。
        render_as_batch=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式：真正连库执行迁移。"""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
