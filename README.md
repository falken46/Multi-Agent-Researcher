# Multi-Agent 研究助手

基于 LangGraph 的 Multi-Agent 自动化研究助手。用户输入研究主题后，系统按 `Planner -> Researcher -> Writer` 的流程生成结构化 Markdown 研究报告。

## 当前实现阶段

当前处于 Phase 6：FastAPI 后端。

已确认的实现边界：

- 使用 LangGraph `StateGraph` 编排 Agent。
- `Planner` / `Researcher` / `Writer` 后续分别放在独立文件中。
- system prompt 后续统一放在 `prompts/*.md`。
- `tools/` 只放外部 IO 工具，例如 `web_search` 和 `web_fetch`。
- PRD 中提到的 `markdown_writer` 不作为独立工具实现，Markdown 报告生成由 `agents/writer.py` 中的 Writer Agent 负责。
- LLM 接口使用 DeepSeek API 的 OpenAI 兼容格式。
- Docker 暂不实施，后续按需进入容器化阶段。

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

复制 `.env.example` 为 `.env`，并填入实际 Key：

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

## 目录结构

```text
agents/      LangGraph 状态、图结构与三个 Agent
backend/     FastAPI API 与 SSE 流式推送
frontend/    Streamlit 前端
prompts/     Agent system prompt
tests/       pytest 测试
tools/       Web Search / Web Fetch 等 IO 工具
```
