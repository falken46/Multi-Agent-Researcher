# Multi-Agent 研究助手

基于 LangGraph 的 Multi-Agent 自动化研究助手。用户输入研究主题后，系统按 `Planner -> Researcher -> Writer` 的流程生成结构化 Markdown 研究报告。

## 当前实现阶段

v1 的 Phase 0—9 已完成；v2 的 **Phase 10 基础设施层已完成，当前等待验收**。下一阶段为 Phase 11 RAG 检索层，尚未开始。

Phase 10 已完成：

- 使用 LangGraph `StateGraph` 编排 Agent。
- `Planner` / `Researcher` / `Writer` 分别位于独立文件，system prompt 统一位于 `prompts/*.md`。
- `tools/` 只放外部 IO 工具，例如 `web_search` 和 `web_fetch`。
- `core/config.py` 集中管理配置，业务代码不再直接读取环境变量。
- `core/llm.py` 是唯一 LLM 调用入口，统一处理超时、重试、token、成本与 trace。
- `core/trace.py` 将任务事件写入 `traces/{date}/{trace_id}.jsonl` 并支持汇总。
- 当前共 61 条离线测试通过，其中保留 v1 原有 41 条回归测试。

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
frontend/    Streamlit 前端
prompts/     Agent system prompt
tests/       pytest 测试
tools/       Web Search / Web Fetch 等 IO 工具
```
