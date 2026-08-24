# Claude Code 工作准则 - DeepResearch Agent

> 本文档是 Claude Code 在协助开发本项目时必须遵守的规则。
> v2 在 v1 五条铁律基础上新增四条（铁律 6—9），原有铁律继续有效。

## 1. 项目核心原则

**本项目的目标是体现"工程化的多智能体系统能力"：框架编排、检索底层、评测工程、可观测性、工程化交付。**

与项目 1 (Mini Claude Code) 的关键区别：

- 项目 1：禁用框架，从 0 实现
- 项目 2（本项目）：**必须使用 LangGraph 框架**

v2 补充定位：本项目须与作者的企业实习经历**互补而非重叠**。实习已覆盖业务场景路由、Prompt 工程、MCP 调用方、受约束数据查询，因此本项目**不再重复应用层调优**，聚焦底层与工程。

---

## 2. 编码铁律

### 🔴 铁律 1：必须使用 LangGraph 编排

- ✅ 所有 Agent 间协作必须通过 LangGraph StateGraph 实现
- ✅ 状态共享必须通过 TypedDict 定义清晰的 schema
- ❌ 禁止用 if-else 手写 Agent 调度逻辑
- ❌ 禁止把所有 Agent 逻辑塞进单个函数

### 🔴 铁律 2：每个 Agent 一个文件

- ✅ Planner / Researcher / Critic / Writer 必须分别在独立文件中
- ✅ 每个 Agent 文件只暴露一个 `xxx_node(state)` 函数
- ❌ 禁止跨文件的 Agent 函数互相直接调用

### 🔴 铁律 3：Prompt 文件化

- ✅ 所有 system prompt 必须放在 `prompts/*.md` 中
- ✅ Python 代码通过 `load_prompt(name)` 函数加载
- ❌ 禁止在 Python 代码中硬编码长 Prompt（>5 行）

### 🔴 铁律 4：工具与 Agent 解耦

- ✅ 工具放在 `tools/`，被 Agent import 调用
- ✅ 工具函数只做 IO，不做 LLM 调用，**不做策略决策**
- ❌ 禁止在工具内部嵌入 Agent 逻辑
- ❌ 禁止 Agent 直接 import `rag/` 内部模块，必须经由 `tools/kb_search.py`

### 🔴 铁律 5：日志可观测

- ✅ 每个 Agent 节点入口必须打印日志
- ✅ 工具调用必须打印输入 / 输出摘要
- ✅ LangGraph 状态变更必须可追溯

### 🔴 铁律 6：LLM 调用唯一入口【v2 新增】

- ✅ 所有 LLM 调用必须经由 `core/llm.py` 的 `chat()` / `achat()`
- ✅ 调用时必须传入 `node` 与 `trace_id`
- ❌ 禁止在 Agent 或工具中直接构造 `OpenAI()` 客户端
- ❌ 禁止绕过 `core/llm.py` 直接调 SDK

> 理由：token 计量、重试策略与 trace 记录全部收敛在这一层。任何绕过都会造成成本数据缺口，进而让评测报告失真。

### 🔴 铁律 7：配置集中管理【v2 新增】

- ✅ 所有配置项在 `core/config.py` 中声明，带类型与默认值
- ✅ 新增配置必须同步更新三处：`core/config.py`、`.env.example`、`TECH_STACK.md`
- ❌ 禁止在业务代码中出现 `os.getenv`

### 🔴 铁律 8：指标只从 trace 取【v2 新增】

- ✅ 所有量化指标必须来自 `traces/*.jsonl` 或评测脚本产出
- ❌ 禁止从日志文本里正则抠数字当指标
- ❌ 禁止在文档、README 或简历材料中写入未真实跑出的数字

> 理由：`EVAL.md` 与 `RESUME_MAPPING.md` 中的数字会直接进入求职材料，必须可复现、可追溯。

### 🔴 铁律 9：检索层测试不依赖网络【v2 新增】

- ✅ `rag/` 相关测试一律使用 `EMBEDDING_BACKEND=fake`
- ✅ 需要真实 API 或网络的测试必须标记 `@pytest.mark.live`
- ❌ 禁止让 CI 依赖任何 API Key

---

## 3. 编码风格规范

### 3.1 Python 风格

- Python 3.10+，全部使用 type hint
- 异步函数明确标注 `async def`
- v2 大量使用 asyncio：并发必须带 `Semaphore` 上限与超时，`gather` 必须 `return_exceptions=True`

### 3.2 命名约定

- Agent 节点函数：`{role}_node(state: ResearchState) -> dict`
- 工具函数：动词开头，如 `web_search`、`kb_search`
- Prompt 文件：`{role}_system.md`
- 检索层模块：按职责单一命名（`splitter` / `hybrid` / `rerank`），不使用 `utils.py` 这类兜底文件

### 3.3 错误处理

- Agent 节点内捕获异常 → 写入 `state["errors"]` 并 emit `error` trace 事件，不抛出
- 工具函数失败 → 抛出明确异常 → 由 Agent 捕获
- 结构化输出解析失败 → **降级**（给中性值）而非中断主流程

---

## 4. 工作流程要求

### 4.1 阶段执行

- 严格按照 `TASKS.md` 顺序（v2 从 Phase 10 开始）
- 每个 Phase 完成后**暂停**，等待用户验收
- 禁止跨阶段实现

### 4.2 修改前必读

- 修改任何文件前，先读当前内容
- 修改 LangGraph 图结构前，先确认 `ARCHITECTURE.md` 第 2.4 节的状态机设计
- 修改检索链路前，先确认 `ARCHITECTURE.md` 第 2.2 节的流水线定义

### 4.3 测试要求

- 每个模块完成后必须有对应测试
- Phase 完成后必须运行 `uv run pytest`
- **回归红线**：v1 原有 41 条测试必须始终通过

---

## 5. 沟通要求

### 5.1 回复格式

- 改动开始前简述"本次要做什么"
- 改动结束后输出"本次改动总结 + 下一步建议"
- 涉及 LangGraph 图结构或检索链路变更时，先用文字描述新结构再写代码

### 5.2 禁止行为

- ❌ 禁止"顺便"重构已通过验收的代码
- ❌ 禁止主动引入新依赖（必须先与作者确认）
- ❌ 禁止生成未运行验证过的代码
- ❌ 禁止在未跑出数据前，向 `README.md` / `EVAL.md` / `RESUME_MAPPING.md` 填写任何指标数字

---

## 6. 关键术语对齐

| 术语 | 含义 |
|------|------|
| StateGraph | LangGraph 的状态机对象 |
| Node | 状态机中的一个 Agent 节点 |
| Conditional Edge | 基于状态判断走向的条件边 |
| ResearchState | 各 Agent 共享的状态 schema |
| Checkpointer | LangGraph 的状态持久化机制，支持断点续跑 |
| trace_id | 一次任务的全链路追踪标识 |
| RRF | Reciprocal Rank Fusion，基于排名的多路召回融合 |
| Rerank | 对召回候选做二次精排 |
| retry_count | 技术失败重试次数 |
| revision_count | 质量不达标的主动返工次数 |
| 降级 (fallback) | 本地召回不足时切换到联网检索 |

---

## 7. 文档同步义务

改动代码时，必须同步更新受影响的文档：

| 改动 | 需同步更新 |
|------|-----------|
| 新增配置项 | `core/config.py`、`.env.example`、`TECH_STACK.md` |
| 改图结构 | `ARCHITECTURE.md` 2.4 节 |
| 改检索链路 | `ARCHITECTURE.md` 2.2 节 |
| 改指标定义 | `EVAL.md` 第 3 节 |
| 跑出新数据 | `README.md`、`EVAL.md` 报告、`RESUME_MAPPING.md` 占位符 |
| 新增依赖 | `TECH_STACK.md`、`pyproject.toml` |

遇到设计分歧，**先与作者确认，不要擅自决定**。
