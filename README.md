# Multi-Agent 研究助手

基于 LangGraph 的 Multi-Agent 自动化研究助手。用户输入研究主题后，系统按 `Planner -> Researcher -> Writer` 的流程生成结构化 Markdown 研究报告。

## 当前实现阶段

v1 的 Phase 0—9 已完成；v2 的 **Phase 10 基础设施层与 Phase 11 RAG 检索层已完成，当前等待 Phase 11 验收**。尚未开始 Phase 12 Agent 编排升级。

当前已完成：

- 使用 LangGraph `StateGraph` 编排 Agent。
- `Planner` / `Researcher` / `Writer` 分别位于独立文件，system prompt 统一位于 `prompts/*.md`。
- `tools/` 只放外部 IO 工具，例如 `web_search` 和 `web_fetch`。
- `core/config.py` 集中管理配置，业务代码不再直接读取环境变量。
- `core/llm.py` 是唯一 LLM 调用入口，统一处理超时、重试、token、成本与 trace。
- `core/trace.py` 将任务事件写入 `traces/{date}/{trace_id}.jsonl` 并支持汇总。
- `rag/` 实现 Markdown / TXT / PDF 加载、中文切分、可插拔 embedding、Chroma 与 BM25 双路召回、RRF 融合和可切换 rerank。
- `tools/kb_search.py` 只封装本地检索 IO；联网降级策略仍留给下一阶段的 Researcher。
- `data/kb/` 提供 20 篇可复现的技术语料，索引运行时产物不进入 Git。
- 当前共 69 条离线测试通过，其中保留 v1 原有 41 条回归测试。

## 本地知识库

首次建库默认使用 `fastembed` 本地 ONNX 模型：

```bash
uv run python -m rag.index_cli --dir data/kb
```

只验证建库流程或运行测试时，可以使用不下载模型的确定性后端：

```bash
uv run python -m rag.index_cli --dir data/kb --embedding-backend fake
```

Chroma 与 BM25 索引默认分别写入 `data/chroma/` 和 `data/bm25/`，两者都属于可重建运行时数据，已被 `.gitignore` 排除。

## 环境配置

本项目使用 `uv` 管理 Python 环境。

```bash
uv sync --group dev
```

激活环境后运行命令：

```bash
uv run python --version
uv run pytest
```

如需使用传统 pip：

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 环境变量

复制 `.env.example` 为 `.env`，并填入实际 Key。完整配置及默认值以 `.env.example` 为准，最小联网运行配置如下：

```bash
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
TAVILY_API_KEY=tvly-xxx
MODEL_NAME=deepseek-v4-flash
SEARCH_PROVIDER=tavily
MAX_RETRY=2
```

Phase 1 之后的联网搜索默认使用 Tavily。请在 `.env` 中配置 `TAVILY_API_KEY`，否则真实搜索会明确报错；`duckduckgo` 仅保留为显式开发备用选项。

## 后端启动

```bash
uv run uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
```

- `GET /health`：健康检查。
- `POST /research`：提交 `{ "topic": "研究主题" }`，以 SSE 方式返回 Agent 进度。

## 前端启动

```bash
uv run streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

默认前端连接 `http://127.0.0.1:8000`。如果后端端口不同, 可在环境变量中设置：

```bash
BACKEND_URL=http://127.0.0.1:8001
```

## 目录结构

```text
agents/      LangGraph 状态、图结构与三个 Agent
backend/     FastAPI API 与 SSE 流式推送
core/        集中配置、LLM 入口、成本换算与 JSONL trace
data/kb/     可被 Git 追踪的知识库源语料
frontend/    Streamlit 前端
prompts/     Agent system prompt
rag/         文档切分、双路索引、RRF 与 rerank 流水线
tests/       pytest 测试
tools/       KB Search / Web Search / Web Fetch 等 IO 工具
```
