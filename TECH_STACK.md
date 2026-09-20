# DeepResearch Agent - 技术栈 (v2)

---

## 1. 运行环境

- **Python**: 3.12（项目 `.python-version` 已锁定；最低要求 3.10+）
- **包管理**: `uv`（项目已使用，`uv.lock` 已提交）
- **OS**: Windows / macOS / Linux
- **Docker**: Docker Engine 24+ / Docker Compose v2.24+（容器化部署；使用 `env_file.required` 与条件依赖）

---

## 2. 依赖分层

### 2.1 v1 已有依赖（保留）

| 包名 | 版本约束 | 用途 |
|------|----------|------|
| openai | >=1.0.0 | DeepSeek API 的 OpenAI 兼容 SDK |
| langgraph | >=0.2.0 | 多智能体状态机编排 |
| langchain-core | >=0.3.0 | LangGraph 依赖 |
| fastapi | >=0.110.0 | 后端 Web 框架 |
| uvicorn | >=0.27.0 | ASGI 运行时 |
| sse-starlette | >=2.0.0 | SSE 流式推送 |
| streamlit | >=1.30.0 | 前端 Demo |
| tavily-python | >=0.3.0 | 联网搜索（默认） |
| duckduckgo-search | >=5.0.0 | 联网搜索（备用） |
| beautifulsoup4 | >=4.12.0 | HTML 解析 |
| requests | >=2.31.0 | HTTP 请求 |
| python-dotenv | >=1.0.0 | 环境变量加载 |
| rich | >=13.0.0 | 终端日志 |

### 2.2 v2 新增依赖

| 包名 | 用途 | 选型说明 |
|------|------|----------|
| **fastembed** | 文本向量化 | ONNX 运行时，**不依赖 torch**；模型 `BAAI/bge-small-zh-v1.5` 约 100MB 级 |
| **chromadb** | 向量数据库 | 嵌入式，持久化到本地目录，零运维 |
| **rank-bm25** | BM25 关键词检索 | 轻量纯 Python 实现，够用且无额外服务 |
| **jieba** | 中文分词 | BM25 通道的前置分词，中文场景必需 |
| **pydantic-settings** | 配置管理 | 类型校验 + `.env` 加载，替代散落的 `os.getenv` |
| **langgraph-checkpoint-sqlite** | 断点续跑 | 已作为直接依赖接入；提供异步 `AsyncSqliteSaver`，状态按 `thread_id` 持久化 |
| **mcp[cli]** | MCP Server | 官方 Python SDK v2；使用 `MCPServer` 暴露 `deep_research` / `kb_search`，CLI 用于本地 stdio 调试 |
| **pypdf** | PDF 解析 | 知识库支持 PDF 文档 |

> `onnxruntime` 由 `fastembed` 间接引入，无需显式声明。

#### Phase 12 编排依赖锁定

`pyproject.toml` 保存允许升级的依赖约束，实际可复现安装版本以 `uv.lock` 为准。当前锁文件中的关键版本为：

| 包名 | `pyproject.toml` 约束 | `uv.lock` 版本 | 关系 |
|------|----------------------|----------------|------|
| `langgraph` | `>=0.2.0` | `1.1.10` | 图编排、v2 streaming 与 durability |
| `langgraph-checkpoint-sqlite` | `>=3.0.0` | `3.1.1` | Phase 12 新增的 SQLite Checkpointer 直接依赖 |
| `langgraph-checkpoint` | 间接依赖 | `4.2.0` | Checkpoint 基础协议与状态模型 |
| `aiosqlite` | 间接依赖 | `0.22.1` | `AsyncSqliteSaver` 的异步 SQLite 驱动 |
| `mcp[cli]` | `>=2.0,<3.0` | `2.1.1` | MCP v2 server/client、stdio transport、工具 schema 与本地调试 CLI |
| `ruff` | `>=0.16.3,<1.0.0` | `0.16.4` | Phase 15 的本地与 CI lint 门禁 |

> 这里记录的是锁文件事实，不把锁版本反写成业务代码判断。依赖升级后应重新生成 `uv.lock`，并先验证流式事件结构、Checkpoint 恢复与完整测试集。

### 2.3 开发依赖

| 包名 | 用途 |
|------|------|
| pytest | 单元测试 |
| pytest-mock | Mock |
| pytest-asyncio | 异步测试（v2 大量使用） |
| pytest-cov | 覆盖率统计 |
| httpx | FastAPI 测试客户端（仅开发依赖；Phase 12 的同步检索工具通过 `asyncio.to_thread` 接入异步节点） |
| ruff | Lint（`E4` / `E7` / `E9` / `F` / `I`），本地与 CI 使用；当前锁定 `0.16.4` |

### 2.4 v3 变更（Phase 17—20 已完成）

> v3 主要升级存储、可观测与接口工程，**不引入新的研究能力**。

**新增**

| 包名 | 用途 | 选型说明 |
|------|------|----------|
| **pymilvus** `3.0.1` | Milvus 客户端 | ✅ 已接入。**诚实前提：本项目 128 个切片 Chroma 完全够用**，换它是技术栈对口不是性能需要，代价是多三个容器（milvus + etcd + minio） |
| **milvus-lite** `3.2.1` | 本地文件形态的 Milvus | ✅ 已接入。让测试与无 Docker 环境跑同一套代码，CI 不需要起完整 Milvus |
| **sqlalchemy** `2.0.52` | ORM | ✅ 已接入。4 张表，方言中立（不用 JSONB / ARRAY），同一套模型 SQLite 与 PostgreSQL 都能跑 |
| **alembic** `1.19.2` | 数据库迁移 | ✅ 已接入。⚠️ `alembic.ini` **必须纯 ASCII** —— 它用 `encoding="locale"` 读文件，中文 Windows 上是 GBK，非 ASCII 字符会在 configparser 里抛 UnicodeDecodeError，报错点离配置很远 |
| **psycopg[binary]** `3.3.5` | PostgreSQL 驱动 | ✅ 已接入 |
| **opentelemetry-sdk** `1.44.0` + **opentelemetry-exporter-otlp-proto-http** | 可观测导出 | ✅ 已接入。**选 OTel 标准而非绑定某家 SDK**：Laminar 是 OTel 原生、Langfuse 也吃 OTLP，换后端只改 endpoint 与 headers。`core/trace.py` 的事件模型一行未改，JSONL 仍是主路径。2026-09-20 已用 Langfuse Cloud 完成一次真实 OTLP Trace 验证，证据见 `docs/langfuse-trace.png`；未使用 Langfuse 专用 SDK |

**移除**

| 包名 | 原因 |
|------|------|
| chromadb | ⚠️ **未移除**：`VECTOR_BACKEND=chroma` 分支保留用于 A/B 对照，验证换库未改变检索结果 |

**配置字段变更**

```bash
# 保留（chroma 后端仍可用，用于 A/B 对照）
CHROMA_DIR=data/chroma
CHROMA_COLLECTION=deepresearch_kb

# 新增
VECTOR_BACKEND=chroma                 # chroma | milvus
# ⚠️ 变量名是 MILVUS_ENDPOINT 不是 MILVUS_URI —— 后者被 pymilvus 自己占用，
#    它在 import 时就读该变量并强制按 http 解析，填本地文件路径会在 import 阶段崩。
MILVUS_ENDPOINT=http://localhost:19530
MILVUS_COLLECTION=deepresearch_kb
MILVUS_TOKEN=
DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/deepresearch
API_KEYS=                             # key1:actor1,key2:actor2；留空关闭鉴权
FRONTEND_API_KEY=                    # Streamlit 使用的单个 key，须命中 API_KEYS
OTEL_ENDPOINT=                        # 留空 = 只写本地 JSONL
OTEL_SERVICE_NAME=deepresearch-agent
OTEL_HEADERS=                         # 形如 Authorization=Bearer xxx
OTEL_TIMEOUT=10
```

> 沿用既有约定：新增配置项必须同时更新三处 —— `core/config.py`、`.env.example`、本文档。

**v3 升级风险**

| 依赖 | 风险点 |
|------|--------|
| pymilvus | 主版本升级会改变 collection schema 与索引参数，需重建索引 |
| alembic | 迁移脚本一旦执行过就不能改写，只能追加新迁移 |
| **迁移顺序** | **先换向量库并保持旧语料**，用同一公开基准完成 A/B 对照；若同时更换语料，出现差异时无法区分适配器问题与数据分布变化 |

---

## 3. 外部服务

| 服务 | 用途 | 计费 | 降级方案 |
|------|------|------|----------|
| DeepSeek API | 所有 LLM 调用 | 按 token | 无（核心依赖） |
| Tavily | 联网搜索 | 免费额度 | DuckDuckGo（显式配置） |
| Embedding | 本地 ONNX 推理 | 免费 | 远程 API 后端可选 |
| Rerank | 本地 ONNX 或 LLM | 免费 / 按 token | 两种实现可切换 |

> **成本可控性说明**：除 LLM 与搜索外，检索链路全部本地推理，不引入额外付费服务。这既是成本考虑，也保证项目在没有额外 API Key 的机器上可复现。

---

## 4. 环境变量（v2 完整版）

> ⚠️ 本节是 **v2** 的完整清单。v3 新增的 `VECTOR_BACKEND` / `MILVUS_*` / `DATABASE_URL` 等见 §2.4，
> 以 `.env.example` 为准。

```bash
# ---- LLM ----
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-v4-flash
LLM_TIMEOUT=60
LLM_MAX_RETRY=3
MODEL_PRICING={"deepseek-v4-flash":{"input_cache_hit":0.02,"input_cache_miss":1.0,"output":2.0},"deepseek-v4-pro":{"input_cache_hit":0.025,"input_cache_miss":3.0,"output":6.0}}
MODEL_PRICING_CURRENCY=CNY
MODEL_PRICING_VERSION=2026-08-24

# ---- 联网搜索 ----
SEARCH_PROVIDER=tavily
TAVILY_API_KEY=tvly-xxx

# ---- 检索层 ----
EMBEDDING_BACKEND=fastembed          # fastembed | remote | fake
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_REMOTE_URL=
EMBEDDING_API_KEY=
EMBEDDING_TIMEOUT=30
CHROMA_DIR=data/chroma
CHROMA_COLLECTION=deepresearch_kb
BM25_INDEX_PATH=data/bm25/index.pkl
KB_DIR=data/kb
CHUNK_SIZE=500
CHUNK_OVERLAP=80
RETRIEVAL_TOP_K=20
VECTOR_SEARCH_ENABLED=true
BM25_SEARCH_ENABLED=true
RERANK_BACKEND=onnx                  # onnx | llm | none
RERANK_MODEL=BAAI/bge-reranker-base
RERANK_TOP_N=5
RRF_K=60
KB_SCORE_THRESHOLD=0.35

# ---- 编排 ----
RESEARCH_CONCURRENCY=3
MAX_RETRY=2                          # 技术失败重试上限
MAX_REVISION=2                       # 质量返工上限
QUALITY_THRESHOLD=0.7
CHECKPOINT_DB=data/checkpoints.sqlite

# ---- 可观测 ----
TRACE_DIR=traces
TRACE_ENABLED=true

# ---- 前端 ----
BACKEND_URL=http://127.0.0.1:8000
FRONTEND_API_KEY=
```

> `.env.example` 需与本节保持同步。新增配置项必须同时更新三处：`core/config.py`、`.env.example`、本文档。

`CHECKPOINT_DB` 保存的是 LangGraph 恢复状态，不是指标数据库。后端在一次 `astream()` 的完整生命周期内持有 `AsyncSqliteSaver`，以稳定 `thread_id` 读写状态；恢复时传入 `None` 继续已有任务，并使用 `durability="sync"` 保证下一步执行前完成持久化。该 SQLite 文件及其 WAL/SHM 文件均为运行时产物，已从 Git 排除。

模型价格默认值按 2026-08-24 的 [DeepSeek 官方模型与价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing) 配置，单位为人民币 / 百万 token；输入成本分别记录缓存命中与未命中。价格变化时必须同时更新 `MODEL_PRICING` 和 `MODEL_PRICING_VERSION`。

---

## 5. 依赖安装

```bash
uv sync --group dev
```

首次使用默认后端会下载 embedding 模型，首次使用 ONNX rerank 时还会下载独立的 cross-encoder 模型，后续从本地缓存加载。只做离线测试时使用 `EMBEDDING_BACKEND=fake` 与 `RERANK_BACKEND=none`，不会下载模型或调用网络。

---

## 6. 版本升级注意

| 依赖 | 风险点 |
|------|--------|
| langgraph | 当前锁定 `1.1.10`；升级可能改变 `astream(version="v2")` 事件结构、节点注入或 durability 行为，升级前需跑图、后端流与 checkpoint 测试 |
| langgraph-checkpoint-sqlite | 当前锁定 `3.1.1`；升级前验证 `AsyncSqliteSaver` 生命周期、同一 `thread_id` 恢复和 `None` 输入续跑 |
| chromadb | 主版本升级会改变持久化目录格式，需重建索引（仅 `VECTOR_BACKEND=chroma` 时相关） |
| pymilvus / milvus-lite | 主版本升级会改变 collection schema 与索引参数，需重建索引；注意环境变量不能叫 `MILVUS_URI` |
| sqlalchemy / alembic | 迁移脚本一旦执行过不可改写，只能追加；`alembic.ini` 必须保持纯 ASCII |
| fastembed | 模型名称随版本调整，锁定 `EMBEDDING_MODEL` 后不随意变更，否则历史向量库失效 |
| mcp | v2 已把 `FastMCP` 更名为 `MCPServer`；升级前验证工具 JSON Schema、结构化输出、stdio 握手与 Claude Code 配置 |
| ruff | 新版本可能新增或调整规则；升级后先运行 `ruff check .`，不要通过全局 ignore 掩盖真实错误 |

> **重要**：更换 embedding 模型等同于让整个向量库失效，必须重建索引并重跑评测，否则评测数据不可比。

---

## 7. Docker 与 CI 交付约束

### 7.1 镜像边界

- `Dockerfile.backend` 同时服务一次性 indexer、Alembic migrate 和 FastAPI 后端；运行阶段安装 `libgomp1` 以支持 ONNX Runtime，并显式复制 `alembic.ini` 与 `db/`。新增运行时包后必须同步镜像复制清单。
- `Dockerfile.frontend` 只复制 `core/` 与 `frontend/` 源码。
- Compose 只向前端注入 `FRONTEND_API_KEY`，不把后端完整的 `API_KEYS` 主体映射暴露给前端容器。
- MinIO 使用官方 Quay 镜像 `quay.io/minio/minio:RELEASE.2024-12-18T13-15-44Z`；Docker Hub 的同标签已无法拉取，真实 Compose 验收时已修正。
- 两个 Dockerfile 均从 `uv.lock` 执行 `uv sync --frozen --no-dev --no-install-project`，最终阶段不保留 uv 二进制和依赖下载缓存。
- 两个运行阶段统一使用 UID/GID `10001` 的非 root 用户 `app`，并提供容器健康检查。

当前 `pyproject.toml` 尚未把前后端运行依赖拆成独立组，因此两个镜像都携带完整运行依赖，实测约 324 MB。当前优先保持单一锁文件和低维护成本；如果进入更严格的生产发布流程，再拆分 frontend/backend dependency group。

### 7.2 Compose 启动与持久化

```text
indexer --service_completed_successfully--> backend --service_healthy--> frontend
```

`chroma-data`、`bm25-data`、`checkpoint-data`、`trace-data` 与 `model-cache` 都是命名卷。命名卷避免把运行时状态打进镜像，并规避 Linux 上宿主机自动创建 bind 目录后非 root 容器无写权限的问题。

### 7.3 CI 供应链与离线边界

- `actions/checkout` 与 `astral-sh/setup-uv` 固定到完整 commit SHA，降低可移动 tag 带来的供应链风险。
- uv 固定为 `0.11.3`，依赖通过 `uv sync --frozen --group dev` 安装。
- CI 只授予 `contents: read`，不引用 GitHub Secret。
- 质量门禁依次运行 `ruff check .` 与 `python -m pytest -m "not live"`；后者避开 Windows 中文路径下 console-script trampoline 的兼容问题，也显式排除真实网络/付费测试。
