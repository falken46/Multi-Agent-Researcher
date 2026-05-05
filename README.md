# Multi-Agent 研究助手

基于 LangGraph 的 Multi-Agent 自动化研究助手。用户输入研究主题后，系统按 `Planner -> Researcher -> Writer` 的流程生成结构化 Markdown 研究报告。

## 当前实现阶段

当前处于 Phase 0：项目初始化与基础环境配置。

已确认的实现边界：

- 使用 LangGraph `StateGraph` 编排 Agent。
- `Planner` / `Researcher` / `Writer` 后续分别放在独立文件中。
- system prompt 后续统一放在 `prompts/*.md`。
- `tools/` 只放外部 IO 工具，例如 `web_search` 和 `web_fetch`。
- PRD 中提到的 `markdown_writer` 不作为独立工具实现，Markdown 报告生成由 `agents/writer.py` 中的 Writer Agent 负责。
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
ANTHROPIC_API_KEY=sk-ant-xxx
TAVILY_API_KEY=tvly-xxx
MODEL_NAME=claude-sonnet-4-6
SEARCH_PROVIDER=tavily
MAX_RETRY=2
```

如果没有 Tavily Key，可将 `SEARCH_PROVIDER` 改为 `duckduckgo`。

## 目录结构

```text
agents/      LangGraph 状态、图结构与三个 Agent
backend/     FastAPI API 与 SSE 流式推送
frontend/    Streamlit 前端
prompts/     Agent system prompt
tests/       pytest 测试
tools/       Web Search / Web Fetch 等 IO 工具
```
