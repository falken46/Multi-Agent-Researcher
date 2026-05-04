# Multi-Agent 研究助手 - 技术栈

## 1. 运行环境

- **Python**: 3.10+
- **OS**: macOS / Linux / Windows (WSL2)
- **Docker**: 20.10+（用于容器化部署）
- **浏览器**: Chrome / Firefox（用于访问 Streamlit）

## 2. 核心依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| openai | >=1.0.0 | DeepSeek API 的 OpenAI 兼容 SDK |
| langgraph | >=0.2.0 | Multi-Agent 状态机框架 |
| langchain-core | >=0.3.0 | LangGraph 依赖 |
| fastapi | >=0.110.0 | 后端 Web 框架 |
| uvicorn | >=0.27.0 | FastAPI 运行时 |
| streamlit | >=1.30.0 | 前端 Web 框架 |
| tavily-python | >=0.3.0 | Web 搜索（默认正式路径） |
| duckduckgo-search | >=5.0.0 | Web 搜索（显式开发备用） |
| beautifulsoup4 | >=4.12.0 | HTML 解析 |
| requests | >=2.31.0 | HTTP 请求 |
| python-dotenv | >=1.0.0 | 环境变量加载 |
| rich | >=13.0.0 | 终端日志美化 |
| sse-starlette | >=2.0.0 | FastAPI SSE 支持 |

## 3. 开发依赖

| 包名 | 用途 |
|------|------|
| pytest | 单元测试 |
| pytest-mock | Mock 工具 |
| pytest-asyncio | 异步测试 |
| httpx | FastAPI 测试客户端 |

## 4. 完整 requirements.txt

```
# Core
openai>=1.0.0
langgraph>=0.2.0
langchain-core>=0.3.0

# Backend
fastapi>=0.110.0
uvicorn>=0.27.0
sse-starlette>=2.0.0

# Frontend
streamlit>=1.30.0

# Tools
tavily-python>=0.3.0
duckduckgo-search>=5.0.0
beautifulsoup4>=4.12.0
requests>=2.31.0

# Utils
python-dotenv>=1.0.0
rich>=13.0.0

# Dev
pytest>=7.0.0
pytest-mock>=3.10.0
pytest-asyncio>=0.21.0
httpx>=0.27.0
```

## 5. 环境变量

`.env` 文件应包含：

```
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
TAVILY_API_KEY=tvly-xxx
MODEL_NAME=deepseek-v4-pro        # 可切换为 deepseek-v4-flash
SEARCH_PROVIDER=tavily            # 或 duckduckgo
MAX_RETRY=2
```

## 6. 本地启动方式

### 方式 A：分别启动（开发模式）

```bash
# 终端 1：启动后端
uvicorn backend.api:app --reload --port 8000

# 终端 2：启动前端
streamlit run frontend/app.py
```

### 方式 B：Docker Compose（生产模式）

```bash
docker-compose up --build
```

启动后访问：
- 前端：http://localhost:8501
- 后端 API：http://localhost:8000/docs

## 7. 模型选择策略

| 场景 | 推荐模型 | 理由 |
|------|----------|------|
| 开发调试 | deepseek-v4-flash | 快速迭代，成本较低 |
| Planner | deepseek-v4-pro | 任务拆解需推理质量 |
| Researcher | deepseek-v4-flash | 摘要任务无需高推理 |
| Writer | deepseek-v4-pro | 长文撰写需要质量 |

支持通过环境变量动态切换。

## 8. 不使用的技术（明确）

| 技术 | 原因 |
|------|------|
| LangChain（旧版） | 已被 LangGraph 替代 |
| Anthropic SDK | 统一使用 DeepSeek API |
| React/Vue 前端 | Demo 阶段 Streamlit 足够 |
| 数据库 | Demo 无需持久化 |
| Redis/MQ | 单用户场景无需 |
| Kubernetes | Docker Compose 足以演示 |

## 9. 关于 DeepSeek API

DeepSeek API 使用 OpenAI 兼容格式：

- `DEEPSEEK_BASE_URL=https://api.deepseek.com`
- `DEEPSEEK_API_KEY=sk-xxx`
- 推荐模型：`deepseek-v4-pro` / `deepseek-v4-flash`

旧模型名 `deepseek-chat` / `deepseek-reasoner` 将于 2026-07-24 弃用，新实现优先使用 `deepseek-v4-*`。

## 10. 关于 Tavily API Key

Tavily 提供免费额度（每月 1000 次搜索），注册地址：https://tavily.com

本项目默认使用 Tavily 进行联网搜索。缺少 `TAVILY_API_KEY` 时，真实搜索会明确失败，避免静默切换到其他搜索源。

DuckDuckGo 仅作为显式开发备用：
```
SEARCH_PROVIDER=duckduckgo
```
（无需 API Key，但稳定性略差，不作为正式验收路径）
