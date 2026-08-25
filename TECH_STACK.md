# DeepResearch Agent - 技术栈 (v2)

---

## 1. 运行环境

- **Python**: 3.12（项目 `.python-version` 已锁定；最低要求 3.10+）
- **包管理**: `uv`（项目已使用，`uv.lock` 已提交）
- **OS**: Windows / macOS / Linux
- **Docker**: 20.10+（容器化部署）

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
| **langgraph-checkpoint-sqlite** | 断点续跑 | LangGraph 官方 SQLite Checkpointer |
| **mcp[cli]** / **fastmcp** | MCP Server | 暴露 `deep_research` / `kb_search` 工具 |
| **pypdf** | PDF 解析 | 知识库支持 PDF 文档 |
| **httpx** | 异步 HTTP / 测试客户端 | 当前用于 FastAPI 测试，Phase 12 异步网络调用复用；同步远程 embedding 使用既有 `requests` |

> `onnxruntime` 由 `fastembed` 间接引入，无需显式声明。

### 2.3 开发依赖

| 包名 | 用途 |
|------|------|
| pytest | 单元测试 |
| pytest-mock | Mock |
| pytest-asyncio | 异步测试（v2 大量使用） |
| pytest-cov | 覆盖率统计 |
| httpx | FastAPI 测试客户端 |
| ruff | Lint + 格式化（CI 使用） |

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
```

> `.env.example` 需与本节保持同步。新增配置项必须同时更新三处：`core/config.py`、`.env.example`、本文档。

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
| langgraph | 0.2 → 0.3 有 API 变更，升级前先跑 `tests/test_graph.py` |
| chromadb | 主版本升级会改变持久化目录格式，需重建索引 |
| fastembed | 模型名称随版本调整，锁定 `EMBEDDING_MODEL` 后不随意变更，否则历史向量库失效 |

> **重要**：更换 embedding 模型等同于让整个向量库失效，必须重建索引并重跑评测，否则评测数据不可比。
