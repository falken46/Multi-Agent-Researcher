# Claude Code 工作准则 - Multi-Agent 研究助手项目

> 本文档是 Claude Code 在协助开发本项目时必须遵守的规则。

## 1. 项目核心原则

**本项目的目标是体现"工程化的 Multi-Agent 系统能力"，强调框架使用、协作编排、工程落地。**

与项目 1 (Mini Claude Code) 的关键区别：
- 项目 1：禁用框架，从 0 实现
- 项目 2（本项目）：**必须使用 LangGraph 框架**

## 2. 编码铁律

### 🔴 铁律 1：必须使用 LangGraph 编排

- ✅ 所有 Agent 间协作必须通过 LangGraph StateGraph 实现
- ✅ 状态共享必须通过 TypedDict 定义清晰的 schema
- ❌ 禁止用 if-else 手写 Agent 调度逻辑
- ❌ 禁止把所有 Agent 逻辑塞进单个函数

### 🔴 铁律 2：每个 Agent 一个文件

- ✅ Planner / Researcher / Writer 必须分别在独立文件中
- ✅ 每个 Agent 文件只暴露一个 `xxx_node(state)` 函数
- ❌ 禁止跨文件的 Agent 函数互相直接调用

### 🔴 铁律 3：Prompt 文件化

- ✅ 所有 system prompt 必须放在 `prompts/*.md` 中
- ✅ Python 代码通过 `load_prompt(name)` 函数加载
- ❌ 禁止在 Python 代码中硬编码长 Prompt（>5 行）

### 🔴 铁律 4：工具与 Agent 解耦

- ✅ 工具放在 `tools/`，被 Agent import 调用
- ✅ 工具函数只做 IO，不做 LLM 调用
- ❌ 禁止在工具内部嵌入 Agent 逻辑

### 🔴 铁律 5：日志可观测

- ✅ 每个 Agent 节点入口必须打印日志（使用 `rich` 或 `logging`）
- ✅ 工具调用必须打印 输入 / 输出摘要
- ✅ LangGraph 状态变更必须可追溯

## 3. 编码风格规范

### 3.1 Python 风格
- Python 3.10+
- 全部使用 type hint
- 异步函数明确标注 `async def`

### 3.2 命名约定
- Agent 节点函数：`{role}_node(state: ResearchState) -> dict`
- 工具函数：动词开头，如 `web_search`, `fetch_page`
- Prompt 文件：`{role}_system.md`

### 3.3 错误处理
- Agent 节点内捕获异常 → 写入 `state["errors"]`，不抛出
- 工具函数失败 → 抛出明确异常 → 由 Agent 捕获

## 4. 工作流程要求

### 4.1 阶段执行
- 严格按照 TASKS.md 顺序
- 每个 Phase 完成后**暂停**，等待用户验收
- 禁止跨阶段实现

### 4.2 修改前必读
- 修改任何文件前，先 read 当前内容
- 修改 LangGraph 图结构前，确认 ARCHITECTURE.md 中的状态机设计

### 4.3 测试要求
- 每个 Agent 完成后必须有对应的 mock 测试
- Phase 完成后必须运行 `pytest`

## 5. 沟通要求

### 5.1 回复格式
- 改动开始前简述"本次要做什么"
- 改动结束后输出"本次改动总结 + 下一步建议"
- 涉及 LangGraph 图结构变更时，先用文字描述新的图结构再写代码

### 5.2 禁止行为
- ❌ 禁止"顺便"重构已通过验收的代码
- ❌ 禁止主动引入新的依赖（必须先与作者确认）
- ❌ 禁止生成未运行验证过的代码

## 6. 关键术语对齐

| 术语 | 含义 |
|------|------|
| StateGraph | LangGraph 的状态机对象 |
| Node | 状态机中的一个 Agent 节点 |
| Edge | 节点间的转移关系 |
| Conditional Edge | 基于状态判断走向的条件边 |
| ResearchState | 三个 Agent 共享的状态 schema |
| Streaming | 通过 SSE 推送中间状态到前端 |

## 7. 参考资料

实现时优先参考：
1. LangGraph 官方文档：https://langchain-ai.github.io/langgraph/
2. LangGraph Multi-Agent 教程
3. FastAPI 流式响应文档：https://fastapi.tiangolo.com/advanced/custom-response/

遇到设计分歧，**先与作者确认，不要擅自决定**。

## 8. 与项目 1 的协同

如果作者在两个项目间切换：
- 进入本项目时，明确切换到"框架使用模式"
- 不要把项目 1 的"从 0 实现"原则带入本项目
- 两个项目共享 Anthropic API Key，但各自独立 venv