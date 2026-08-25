# DeepResearch Agent

基于 LangGraph 的工程化多智能体研究助手：本地混合检索不足时联网，Critic 定向补查，并以 SSE、SQLite Checkpoint 和任务级 JSONL trace 支撑可观测、可恢复的 Markdown 报告生成流程。

> 当前代码基线已实现 Phase 10—12（M1 核心能力）；Phase 13 已完成公开数据上的 R1—R4 检索消融，并接通 P/Q 编排 runner，但尚未执行付费端到端任务。本文只填写已由结构化 raw 复算的检索数字，不提前填写质量提升率或真实任务加速比。

## 核心能力

- **四 Agent 状态机**：Planner 拆题，Researcher 检索与归纳，Critic 评分并指出缺口，Writer 汇总成报告。
- **本地优先的混合检索**：Chroma 向量召回与 BM25 关键词召回经 RRF 融合，再按配置执行 rerank；本地检索失败或最高分低于配置阈值时才联网补查。
- **受控并发与失败隔离**：Researcher 使用 `asyncio.gather`、`Semaphore`、超时和 `return_exceptions=True` 并发处理独立子问题。
- **有边界的质量返工**：Critic 只让 Researcher 补查明确缺口，并通过返工上限和分数停滞检测防止死循环。
- **可观测、可恢复**：JSONL trace 记录节点、token、估算成本、耗时、降级和返工事件；SQLite Checkpoint 通过稳定 `thread_id` 支持 API 从最近 checkpoint 恢复。
- **离线可回归**：默认测试不依赖 API Key 或网络，检索测试使用确定性 fake embedding。

## 系统架构

```text
用户
 │ 研究主题
 ▼
Streamlit 前端
 │ POST /research（SSE）
 ▼
FastAPI 后端 ───────────────► JSONL trace
 │                            节点 / token / 估算成本 / 耗时 / 返工 / 降级
 │ ResearchState + thread_id
 ▼
LangGraph StateGraph ◄──────► AsyncSqliteSaver（SQLite Checkpoint）
 │                            状态持久化 / API 从最近 checkpoint 恢复
 │
 └─ START → Planner → Researcher
                         ├─ 有结果 → Critic
                         │             ├─ 低分 + 有缺口 + 未达上限 / 未停滞
                         │             │       └─→ Researcher（定向补查）
                         │             └─ 通过 / 无缺口 / 达上限 / 分数停滞 → Writer
                         ├─ 无结果且可重试 → Researcher（技术重试）
                         └─ 无结果且重试耗尽 → Writer（降级报告）

上述所有 Writer 分支 → END

离线建库：data/kb → Loader → Splitter
                              ├─ Embedding → Chroma
                              └─ jieba → BM25 Index

在线查询：Researcher → tools/kb_search.py → Chroma + BM25 → RRF → 可选 Rerank
               └─ 本地检索失败或分数不足 → tools/web_search.py

Planner / Researcher / Critic / Writer
 └─ core/llm.py → 统一模型调用、重试、token 与估算成本记录
```

图中的两类持久化职责不同：Checkpoint 保存 LangGraph 状态，用来继续未完成任务；trace 保存运行事件，用来排查问题和汇总用量，二者不能互相替代。

## 快速开始

### 1. 准备环境

项目声明支持 Python 3.10+，仓库当前使用 Python 3.12，并通过 `uv` 管理依赖。

```bash
uv sync --group dev
uv run python --version
```

### 2. 先跑离线测试

```bash
uv run pytest
```

测试会 mock 外部 LLM / 搜索服务，并对检索使用 fake embedding，不需要 API Key。

### 3. 配置真实运行环境

复制 `.env.example` 为 `.env`：

```powershell
Copy-Item .env.example .env
```

默认配置会在本地证据不足时自动调用 Tavily。为保证真实任务完整运行，请同时填写 DeepSeek 与 Tavily Key；只有明确验证本地高分命中路径时，才可以暂不配置 Tavily。

```dotenv
DEEPSEEK_API_KEY=sk-xxx
TAVILY_API_KEY=tvly-xxx

EMBEDDING_BACKEND=fastembed
RERANK_BACKEND=onnx
TRACE_ENABLED=true
```

完整配置及默认值见 [`.env.example`](.env.example)。首次使用 FastEmbed 或 ONNX rerank 时可能需要下载本地模型。

### 4. 构建本地知识库

```bash
uv run python -m rag.index_cli --dir data/kb
```

默认索引写入 `data/chroma/` 和 `data/bm25/`，它们是可重建的运行时数据，不进入 Git。建库结束后应确认输出中的向量数量与 BM25 数量都大于 0。

> `--embedding-backend fake` 只覆盖当次建库命令，适合隔离测试，不代表应用运行时也会切换为 fake。建库与查询必须使用相同的 embedding 后端和模型；切换后请更新 `.env` 并重新执行上述建库命令，否则可能出现维度不匹配或检索结果失真。

### 5. 启动后端与前端

终端 1：

```bash
uv run uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
```

终端 2：

```bash
uv run streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

浏览器打开 `http://127.0.0.1:8501`。后端健康检查为 `GET /health`，交互式接口文档位于 `http://127.0.0.1:8000/docs`。

## API、SSE 与断点恢复

新任务最安全的做法是只传主题，让后端生成唯一 `thread_id`。以下命令适用于 Bash / Git Bash：

```bash
curl -N -X POST http://127.0.0.1:8000/research \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"topic":"多智能体系统如何控制反思循环？"}'
```

Windows PowerShell 5 / 7 可使用 `curl.exe`；示例使用英文主题以避开旧版 PowerShell 的管道编码差异：

```powershell
'{"topic":"LangGraph reflection loop"}' | curl.exe -N -H "Content-Type: application/json" -H "Accept: text/event-stream" --data-binary "@-" http://127.0.0.1:8000/research
```

请保存 `start` 事件返回的 ID。恢复未完成任务时，使用相同主题和 ID；每个新任务都应使用新的 ID，只有恢复时才复用旧 ID。下面的 `THREAD_ID_FROM_START` 需要替换成该事件返回的真实值：

```json
{
  "topic": "多智能体系统如何控制反思循环？",
  "thread_id": "THREAD_ID_FROM_START",
  "resume": true
}
```

当前恢复入口只在后端 API 提供，Streamlit 页面尚无恢复控件。`resume=true` 必须携带 `thread_id`，且主题必须与已保存任务完全一致。

SSE 可能发送以下事件：

| 事件 | 含义 |
|---|---|
| `start` | 返回本次 `trace_id` 与 `thread_id` |
| `progress` | 节点完成后的状态增量与执行结果 |
| `critic_start` / `critic_done` | 评审开始及评分、评语、缺口 |
| `revision` | Critic 决定定向返工 |
| `fallback` | 本地检索失败或分数不足，切换联网搜索 |
| `usage` | trace 汇总的 token、估算成本、调用次数与耗时 |
| `complete` | 返回最终状态与报告 |
| `error` | 返回流式执行或 checkpoint 恢复错误 |

评审、返工和降级事件由实际路由决定，可能不出现或重复出现。当前 SSE 传输的是节点级进度和自定义事件，不是模型 token 的逐字流式输出。

## 关键配置

| 配置 | 作用 |
|---|---|
| `KB_SCORE_THRESHOLD` | 本地检索低于该分数时触发联网降级 |
| `RESEARCH_CONCURRENCY` | Researcher 同时处理的子问题上限 |
| `MAX_RETRY` | 完全没有研究结果时的技术重试上限 |
| `MAX_REVISION` | Critic 触发质量返工的上限 |
| `QUALITY_THRESHOLD` | Critic 判定通过的分数阈值 |
| `CHECKPOINT_DB` | SQLite checkpoint 文件路径 |
| `TRACE_ENABLED` / `TRACE_DIR` | 是否记录 trace 及其输出目录 |

演示和正式评测应保持 `TRACE_ENABLED=true`，否则无法从 trace 汇总完整的 token、成本与耗时证据。

## 当前验证证据

当前分支全量离线回归结果为：

```text
129 passed
```

这组测试覆盖配置、统一 LLM 入口、trace、检索流水线、公开数据转换、R1—R4 runner、报告重算、四个 Agent、反思回环、并发边界、SSE、前端状态和关闭后重开的 SQLite 恢复。它证明实现满足这些确定性场景，**不等于**真实模型准确率、联网稳定性或生产性能；真实检索数字来自单独保存的结构化评测 raw。

| 维度 | 当前已有证据 | 现在可以得出的结论 | Phase 13 待补 |
|---|---|---|---|
| 功能回归 | 129 项离线测试 | 约定的本地场景可重复通过 | 真实 LLM / 搜索端到端完成率 |
| 检索链路 | 100 题公开 qrels、1,664 passage、400 条结构化观测 | RRF 提升 MRR，但当前 ONNX rerank 产生负收益 | 产品知识库外推与 reranker 模型匹配分析 |
| 并发研究 | fake IO + P1/P2 固定任务 runner 测试 | 并发上限、配对任务和续跑边界生效 | 相同真实任务的串行 / 并行耗时对照 |
| Critic 回环 | 确定性图场景 + Q1/Q2 runner 测试 | 返工路由与 Critic 独立开关生效 | 15 题两轮的完成率与质量变化 |
| 成本观测 | LLM / trace 汇总测试 | token 与配置价格估算链路可追踪 | 固定评测集上的平均 token、成本和耗时 |

Phase 13 已接入公开中文 `C-MTEB/T2Reranking`：固定抽取 100 个 query，将 positive 与 hard negative 合并为 1,664 个 passage 的共享池，并直接沿用公开 qrels。正式 R 轨生成了 400 条结构化观测：RRF（R3）相较向量基线（R1）把 MRR@5 从 **0.7238** 提到 **0.7777**，但 Candidate Recall@20 从 **93.64%** 降至 **92.27%**；ONNX rerank（R4）又把 MRR@5 降至 **0.7377**。因此当前结论是“RRF 改善首个相关结果的平均位置，但并非所有指标都提升；该 reranker 在此数据/候选分布上不应视为有效优化”。完整证据与边界见 [`eval/reports/comparison.md`](eval/reports/comparison.md) 和 [`EVAL.md`](EVAL.md)。

## 技术决策摘要

| 决策 | 为什么这样做 | 代价 / 边界 |
|---|---|---|
| LangGraph 条件图 | 显式表达节点、循环、条件边与恢复点 | 引入框架概念和状态 schema 维护成本 |
| Chroma + BM25 + RRF | 同时覆盖语义相近与关键词精确匹配，RRF 无需直接比较异构分数 | 需要维护两套索引并用评测校准参数 |
| 本地优先、低分联网 | 优先利用可控语料，仅在证据不足时承担联网成本 | 阈值目前仍需正式评测校准 |
| `asyncio` + `Semaphore` | 并发独立 IO 子问题，同时限制资源和 API 压力 | 必须配套超时、限流与异常隔离 |
| Critic 定向返工 | 将“研究证据不足”转换为可检索的明确缺口 | Critic 分数不是第三方质量结论，需对照实验验证收益 |
| SQLite Checkpoint + JSONL trace | 分别处理状态恢复与事件审计 | 产生两类运行时数据，当前更适合单机演示 |
| 统一 LLM Gateway | 集中重试、计量、价格估算和 trace | 公共入口成为需要重点测试的关键模块 |
| 离线 fake embedding | 自动化回归不依赖网络、模型下载和 API Key | 只能验证确定性逻辑，不能代表真实检索质量 |

## 目录结构

```text
agents/       ResearchState、LangGraph 图与四个 Agent 节点
backend/      FastAPI、异步 SSE 与 checkpoint 恢复入口
core/         集中配置、LLM Gateway、成本估算、trace、checkpoint
data/kb/      Git 可追踪的本地知识库源语料
eval/         公开检索集转换、端到端题集、指标与报告
frontend/     Streamlit 任务进度与报告页面
prompts/      四个 Agent 与 LLM reranker 的 system prompt
rag/          加载、切分、embedding、双路召回、RRF 与 rerank
tests/        不依赖外部 API 的 pytest 回归测试
tools/        KB Search / Web Search / Web Fetch 等 IO 工具
```

## 文档导航

- [`ARCHITECTURE.md`](ARCHITECTURE.md)：分层架构、状态机与后续演进设计（其中部分目录属于未来 Phase）。
- [`TECH_STACK.md`](TECH_STACK.md)：技术选型、配置来源和运行约束。
- [`TASKS.md`](TASKS.md)：Phase 10—16 的实施顺序与验收标准。
- [`EVAL.md`](EVAL.md)：Phase 13 评测设计、R 轨真实结果与 P/Q 轨执行边界。
- [`RESUME_MAPPING.md`](RESUME_MAPPING.md)：代码证据与简历表达草案，量化占位符需等待真实评测回填。

## 已知边界与后续路线

- 当前标准测试不调用真实 DeepSeek、Tavily 或远程 embedding，联网效果仍需 live 验证。
- 引用由工作流和 Prompt 组织，尚未实现“证据是否语义支持结论”的自动事实核验。
- SQLite checkpoint 面向单机演示；多实例部署需要外部共享存储。
- Streamlit 尚未提供 checkpoint 恢复 UI，当前需通过 API 恢复。
- Phase 13 将补齐固定数据集、可归因的消融实验和可复现报告；Phase 14、15 的 MCP、Docker 与 CI 尚未实现。
