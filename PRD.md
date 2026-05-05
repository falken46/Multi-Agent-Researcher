# Multi-Agent 自动化研究助手 - 产品需求文档 (PRD)

## 1. 项目背景

本项目是基于 LangGraph 框架构建的 Multi-Agent 协作系统，模拟人类"研究小组"的工作流程，让 AI 自动完成"主题规划 → 资料检索 → 内容撰写"的端到端研究任务。目的是深度掌握 LangGraph 状态机、多 Agent 协作、Function Calling 与工程落地全流程。

## 2. 项目目标

### 2.1 核心目标
- 构建一个 Web 化的 Multi-Agent 研究助手，用户输入研究主题，系统自动产出结构化研究报告
- 完整实现 Planner / Researcher / Writer 三角色协作流程
- 提供可交互的 Web Demo（FastAPI 后端 + Streamlit 前端）
- 项目可容器化部署（Docker），具备生产级工程基础

### 2.2 非目标（明确不做）
- ❌ 不做用户系统与登录认证（单用户 Demo）
- ❌ 不做生产级数据持久化（任务结束即清理）
- ❌ 不做付费模型计量与额度管理
- ❌ 不做超长报告（输出控制在 1500-3000 字内）
- ❌ 不做完整 RAG 系统（仅做轻量网页检索）

## 3. 用户场景

### 3.1 典型用例

用户在 Streamlit 前端输入：
> "请帮我研究'2025 年 AI Agent 领域的主要技术趋势'"

系统自动执行：
1. **Planner Agent** 拆解主题为 5 个子问题
2. **Researcher Agent** 针对每个子问题调用 Web Search 工具
3. **Writer Agent** 根据检索结果撰写结构化 Markdown 报告
4. 前端实时展示各 Agent 的工作进度
5. 最终输出可下载的 Markdown 报告

### 3.2 典型工作流示意

```
用户输入主题
     ↓
[Planner] 输出 5 个研究子问题
     ↓
[Researcher] 并行检索 5 个子问题 → 输出原始资料
     ↓
[Writer] 整合资料 → 输出 Markdown 报告
     ↓
前端展示 + 下载
```

## 4. 功能需求

### 4.1 核心功能（必须实现）

| 功能 | 优先级 | 说明 |
|------|--------|------|
| 主题输入与任务启动 | P0 | Streamlit 前端表单 |
| Planner Agent | P0 | 拆解主题为子问题 |
| Researcher Agent | P0 | 调用 Web Search 收集资料 |
| Writer Agent | P0 | 整合资料生成报告 |
| LangGraph 状态机 | P0 | 三角色协作编排 |
| 工具调用（Web Search） | P0 | Tavily / DuckDuckGo |
| 实时进度展示 | P1 | 流式输出每个 Agent 状态 |
| 报告 Markdown 下载 | P1 | 前端按钮触发 |
| 失败重试机制 | P1 | LangGraph 条件边实现 |

### 4.2 工具集（必须实现）

| 工具名 | 用途 | 调用方 |
|--------|------|--------|
| web_search | 关键词检索网页摘要 | Researcher |
| web_fetch | 抓取指定 URL 完整内容 | Researcher |

说明：Markdown 报告生成由 Writer Agent 负责，不单独实现 `markdown_writer` 工具。`tools/` 仅承载外部 IO 工具，避免工具层嵌入 LLM 写作逻辑。

### 4.3 三个 Agent 的职责定义

#### Planner Agent
- **输入**：用户原始研究主题（字符串）
- **输出**：3-5 个具体子问题（JSON 列表）
- **核心 Prompt 策略**：Few-shot + 结构化输出约束

#### Researcher Agent
- **输入**：单个子问题
- **输出**：检索到的原始资料摘要（含来源 URL）
- **工具调用**：web_search → web_fetch（按需）
- **设计要点**：每个子问题独立运行，可并行

#### Writer Agent
- **输入**：所有子问题 + 对应资料
- **输出**：完整 Markdown 研究报告
- **结构要求**：标题 / 摘要 / 各小节 / 参考来源

## 5. 非功能需求

| 维度 | 要求 |
|------|------|
| 响应时间 | 单次研究任务（5 子问题）< 3 分钟 |
| 并发能力 | Demo 阶段单用户即可 |
| 错误恢复 | 单个子问题失败不影响整体任务 |
| 部署方式 | 支持 Docker 一键启动 |
| 可观测性 | 终端日志 + 前端进度展示 |

## 6. 验收标准

项目完成的判断依据：

- [ ] 前端能正常输入主题并启动任务
- [ ] 三个 Agent 能完整协作走完一次完整流程
- [ ] LangGraph 状态机能正确处理失败与重试
- [ ] 输出的 Markdown 报告结构完整、内容相关
- [ ] 整个项目可通过 `docker-compose up` 启动
- [ ] 至少在 5 个不同主题上跑通端到端流程（参考 TESTING.md）
