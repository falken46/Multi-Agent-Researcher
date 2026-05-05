# Multi-Agent 研究助手 - 架构设计文档

## 1. 整体架构

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit 前端                      │
│        (主题输入 / 进度展示 / 报告下载)               │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP (streaming)
┌──────────────────────▼──────────────────────────────┐
│                  FastAPI 后端                        │
│           (任务接收 / SSE 推送 / 报告返回)            │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│               LangGraph 状态机                       │
│      ┌──────────┐    ┌──────────┐    ┌────────┐     │
│      │ Planner  │───▶│Researcher│───▶│ Writer │     │
│      └──────────┘    └────┬─────┘    └────────┘     │
│                           │                          │
│                      条件边: 失败重试                 │
└───────────────────────────┴──────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │   Tools (Web Search等)   │
              └──────────────────────────┘
```

## 2. 核心模块

### 2.1 LangGraph 状态机 (`agents/graph.py`)

**职责**：定义三 Agent 协作流程，管理共享状态。

**核心状态结构**：
```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

class ResearchState(TypedDict):
    topic: str                          # 用户输入主题
    sub_questions: list[str]            # Planner 输出
    research_results: dict[str, str]    # Researcher 输出 {子问题: 资料}
    final_report: str                   # Writer 输出
    errors: list[str]                   # 错误日志
    retry_count: int                    # 重试计数
```

**状态机定义**：
```python
graph = StateGraph(ResearchState)
graph.add_node("planner", planner_node)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "researcher")
graph.add_conditional_edges(
    "researcher",
    should_retry,  # 判断函数
    {"retry": "researcher", "continue": "writer"}
)
graph.add_edge("writer", END)
```

**关键设计决策**：
- 使用 TypedDict 定义状态，便于 type hint 与调试
- Researcher 节点支持失败重试（最多 2 次）
- Writer 不重试（失败直接报错给前端）

### 2.2 Planner Agent (`agents/planner.py`)

**职责**：将用户主题拆解为 3-5 个具体子问题。

**核心实现**：
```python
def planner_node(state: ResearchState) -> ResearchState:
    """
    输入主题,输出子问题列表。
    使用 Claude API 的结构化输出确保 JSON 格式。
    """
    response = client.messages.create(
        model=MODEL,
        system=PLANNER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": state["topic"]}],
    )
    sub_questions = parse_json_list(response.content[0].text)
    return {"sub_questions": sub_questions}
```

**Prompt 设计要点**：
- Few-shot 示例 2-3 个
- 明确输出格式约束："仅输出 JSON 数组，不要任何解释"
- 限制子问题数量：3-5 个

### 2.3 Researcher Agent (`agents/researcher.py`)

**职责**：针对每个子问题进行网页检索与摘要。

**核心实现**：
```python
def researcher_node(state: ResearchState) -> ResearchState:
    """
    遍历每个子问题,调用 web_search 获取资料。
    支持并行（asyncio.gather）。
    """
    results = {}
    for question in state["sub_questions"]:
        try:
            search_results = web_search_tool(question, max_results=3)
            summary = summarize_results(question, search_results)
            results[question] = summary
        except Exception as e:
            state["errors"].append(f"Q: {question} | {str(e)}")
    return {"research_results": results}
```

**工具调用策略**：
- web_search 返回 top 3 条结果
- 每条结果取 title + snippet + url
- 用 LLM 对结果做一次摘要压缩

### 2.4 Writer Agent (`agents/writer.py`)

**职责**：整合所有研究资料，生成结构化 Markdown 报告。

**报告模板**：
```markdown
# {主题}

## 摘要
{2-3 句话总结}

## {子问题 1}
{基于资料的回答}

## {子问题 2}
...

## 参考来源
- [来源 1](url)
- [来源 2](url)
```

**核心实现**：
```python
def writer_node(state: ResearchState) -> ResearchState:
    """
    将 sub_questions + research_results 拼接为 Prompt,
    调用 Claude 生成 Markdown 报告。
    """
    prompt = build_writer_prompt(
        topic=state["topic"],
        questions=state["sub_questions"],
        results=state["research_results"],
    )
    response = client.messages.create(
        model=MODEL,
        system=WRITER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return {"final_report": response.content[0].text}
```

### 2.5 工具层 (`tools/`)

#### web_search.py
- 使用 Tavily API（免费额度足够）或 DuckDuckGo（无需 Key）
- 返回结构化结果列表

#### web_fetch.py
- 使用 `requests` + `beautifulsoup4`
- 提取页面纯文本，限制长度 ≤ 5000 字符

### 2.6 后端服务 (`backend/api.py`)

**职责**：FastAPI 接口层，处理任务请求与流式推送。

**核心接口**：
```python
@app.post("/research")
async def research(request: ResearchRequest) -> StreamingResponse:
    """
    接收主题,启动 LangGraph,通过 SSE 流式推送进度。
    """
    return StreamingResponse(
        stream_research_progress(request.topic),
        media_type="text/event-stream",
    )
```

### 2.7 前端 (`frontend/app.py`)

**职责**：Streamlit 用户界面。

**核心组件**：
- 主题输入框
- 启动按钮
- 进度展示区（按 Agent 分块展示）
- 最终报告区（Markdown 渲染）
- 下载按钮

## 3. 数据流

### 3.1 完整任务数据流
```
用户输入 topic
   ↓
FastAPI 接收 → 创建 ResearchState(topic=...)
   ↓
LangGraph 启动 → planner_node
   ↓ state.sub_questions = [...]
researcher_node
   ↓ state.research_results = {...}
should_retry? 
   ↓ continue
writer_node
   ↓ state.final_report = "..."
END → 返回完整状态
   ↓
SSE 推送 → Streamlit 渲染
```

### 3.2 失败重试数据流
```
researcher_node 失败 → state.retry_count += 1
   ↓
should_retry: retry_count < 2 ?
   ├─ Yes → 回到 researcher_node
   └─ No → 继续 writer_node（带错误标记）
```

## 4. 目录结构

```
multi-agent-research/
├── docker-compose.yml          # 一键启动配置
├── Dockerfile.backend          # 后端镜像
├── Dockerfile.frontend         # 前端镜像
├── requirements.txt
├── .env.example
├── README.md
│
├── agents/
│   ├── __init__.py
│   ├── graph.py               # LangGraph 状态机定义
│   ├── planner.py             # Planner Agent
│   ├── researcher.py          # Researcher Agent
│   ├── writer.py              # Writer Agent
│   └── state.py               # TypedDict 状态定义
│
├── tools/
│   ├── __init__.py
│   ├── web_search.py          # 网页搜索
│   └── web_fetch.py           # 网页抓取
│
├── prompts/
│   ├── planner_system.md
│   ├── researcher_system.md
│   └── writer_system.md
│
├── backend/
│   ├── __init__.py
│   ├── api.py                 # FastAPI 入口
│   └── streaming.py           # SSE 流式推送
│
├── frontend/
│   └── app.py                 # Streamlit 入口
│
└── tests/
    ├── test_planner.py
    ├── test_researcher.py
    ├── test_writer.py
    └── test_e2e.py            # 端到端测试
```

## 5. 关键技术决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Agent 框架 | LangGraph | 当前 Multi-Agent 编排事实标准 |
| LLM API | Anthropic Claude | 统一使用，工具调用稳定 |
| Web Search | Tavily（备选 DuckDuckGo） | 免费额度足够 Demo |
| 后端 | FastAPI | 异步原生支持 + SSE 流式 |
| 前端 | Streamlit | 快速 Demo，无需前端工程 |
| 部署 | Docker Compose | 工程化标准 |
| 状态存储 | LangGraph 内存状态 | Demo 无需持久化 |

## 6. 关键 Prompt 设计原则

1. **结构化输出强约束**：Planner 输出 JSON、Writer 输出 Markdown
2. **Few-shot 优先于纯指令**：每个 Agent 至少 1 个示例
3. **角色边界清晰**：Researcher 不写报告、Writer 不做检索
4. **错误处理嵌入 Prompt**：明确告知 Agent "如果资料不足，可以标注"