# DeepResearch Agent - 任务拆解清单 (v2)

> Phase 0—9 为 v1，已全部完成（详见 git 历史）。
> v2 从 Phase 10 开始。每个 Phase 完成后暂停验收，并单独 commit。

---

## 进度总览

| Phase | 内容 | 里程碑 | 状态 |
|-------|------|--------|------|
| 10 | 基础设施层（config / llm / trace / costs） | M1 | ✅ 已完成 |
| 11 | RAG 检索层 | M1 | ✅ 已完成 |
| 12 | Agent 编排升级（Critic / 并行 / Checkpoint） | M1 | ✅ 功能完成 |
| 13 | 评测体系 | M2 | 🟨 进行中 |
| 14 | MCP Server | M3 | ⬜ |
| 15 | 工程化（Docker / CI） | M2 | ⬜ |
| 16 | 交付物（README / 架构图 / 简历映射 / 口述稿） | M1—M3 | 🟨 部分完成 |

**里程碑定义**

- **M1 可投最低线**：Phase 10—12 + Phase 16 的 README 部分完成 → GitHub 上线，简历可挂链接
- **M2 完整线**：+ Phase 13、15 → 有评测数据与工程化证据
- **M3 加分线**：+ Phase 14 → MCP Server 与 Demo 录制

---

## Phase 10: 基础设施层

> 目标：把散落的配置、LLM 调用与日志收敛成三个可复用模块，为后续所有 Phase 提供地基。

- [x] T10.1 `core/config.py`：pydantic-settings 定义全部配置项，带默认值与类型校验
- [x] T10.2 `core/costs.py`：模型价格表（从配置读取）+ `estimate()` 换算函数
- [x] T10.3 `core/trace.py`：`new_trace_id()` / `emit(event)` / `summarize(trace_id)`，JSONL 落盘
- [x] T10.4 `core/llm.py`：统一 `chat()` / `achat()`，含超时、指数退避重试、token 计量、自动写 trace
- [x] T10.5 改造 `agents/planner.py`、`agents/researcher.py`、`agents/writer.py` 改用 `core.llm`
- [x] T10.6 更新 `.env.example`
- [x] T10.7 `tests/test_core_llm.py`、`tests/test_core_trace.py`

**验收标准**

- 三个 Agent 不再各自构造 OpenAI client，也不再直接读环境变量
- 跑一次任务后，`traces/` 下生成 JSONL，`summarize()` 能算出总 token 与总耗时
- 原有 41 条测试仍全部通过（**回归红线**）

---

## Phase 11: RAG 检索层

> 目标：补齐"自建检索层"这一最大能力缺口。本 Phase 是整个 v2 的技术核心。

- [x] T11.1 `rag/loader.py`：`.md` / `.txt` / `.pdf` 解析，失败不中断建库
- [x] T11.2 `rag/splitter.py`：递归字符切分，中文标点优先，携带 chunk 元数据
- [x] T11.3 `rag/embeddings.py`：可插拔后端（`fastembed` / `remote` / `fake`）
- [x] T11.4 `rag/vectorstore.py`：Chroma 封装，只暴露 `add` / `query` / `count`
- [x] T11.5 `rag/bm25.py`：jieba 分词 + BM25 索引，与向量库共享 chunk id
- [x] T11.6 `rag/hybrid.py`：RRF 融合
- [x] T11.7 `rag/rerank.py`：ONNX cross-encoder 与 LLM rerank 两种实现，可切换
- [x] T11.8 `rag/pipeline.py`：`build_index()` 与 `search()` 统一入口
- [x] T11.9 `rag/index_cli.py`：命令行建库
- [x] T11.10 `tools/kb_search.py`：封装为工具，返回 `hits` 与 `max_score`
- [x] T11.11 准备知识库语料（`data/kb/`，20—40 篇技术文档）
- [x] T11.12 `tests/test_rag_*.py`：切分、融合、检索各一组测试（使用 `fake` embedding 后端保证确定性）

**验收标准**

- `python -m rag.index_cli --dir data/kb` 能成功建库并输出切片数量
- `kb_search("某个语料内明确存在的问题")` 能返回正确文档，且分数明显高于无关问题
- 单独关闭向量通道或 BM25 通道，检索仍可运行（可降级）
- 切分与融合逻辑有确定性测试，不依赖网络

---

## Phase 12: Agent 编排升级

> 目标：把线性流程升级为带反思回环的并行状态机。

- [x] T12.1 `agents/state.py` 扩展：citations / critique / quality_score / quality_history / missing_aspects / revision_count / trace_id / Usage / fallback_queries
- [x] T12.2 `prompts/critic_system.md`：含明确评分锚点与 JSON schema 说明
- [x] T12.3 `agents/critic.py`：结构化输出 + 解析失败降级
- [x] T12.4 `agents/researcher.py` 异步化：同步工具 `to_thread` + `asyncio.gather` + `Semaphore` + `wait_for` + `return_exceptions=True`
- [x] T12.5 Researcher 双通道：`kb_search` 优先，`max_score` 低于阈值降级 `web_search`，降级事件写 trace
- [x] T12.6 Researcher 返工模式：`missing_aspects` 非空时只查缺口
- [x] T12.7 `agents/graph.py`：新增 critic 节点与 `should_revise` 条件边
- [x] T12.8 防死循环三重保险（硬上限 / 定向补查 / 分数无提升即退出）
- [x] T12.9 接入 `AsyncSqliteSaver` Checkpointer：稳定 `thread_id`、`None` 输入恢复、`durability="sync"`
- [x] T12.10 `backend/api.py` / `backend/streaming.py`：异步 SSE、`updates` + `custom` 事件与恢复参数
- [x] T12.11 `frontend/app.py` 展示 Critic、定向返工、联网降级与 trace 汇总 usage
- [x] T12.12 `tests/test_critic.py`、`tests/test_graph_revision.py`、`tests/test_checkpoint.py`（mock LLM，验证回环、上限、停滞退出与恢复）

**验收标准**

- [x] 构造一个低质量场景，能观察到 Critic 打低分 → 回退 Researcher → 分数提升 → 进入 Writer
- [x] 构造一个永远不达标的场景，验证回退次数**严格不超过** `MAX_REVISION`
- [ ] 并行前后耗时对比有真实任务实测数据并写入 Phase 13 评测报告（Phase 12 已有 fake IO + trace latency 的确定性回归测试，但它不是对外指标）
- [x] 中断任务后可从 Checkpoint 恢复

> ⭐ **Phase 12 完成即达 M1 可投最低线**，此时应立即执行 Phase 16 的 README 部分并推送 GitHub，不要等后续 Phase。

---

## Phase 13: 评测体系

> 目标：产出简历上所有量化数字。详细方案见 `EVAL.md`。

- [x] T13.1 接入公开 `C-MTEB/T2Reranking` 检索集，并构造 15 题端到端编排集
- [x] T13.2 用公开 positive / hard negative / qrels 生成共享 passage 池，替代人工 gold 标注
- [x] T13.3 `eval/metrics.py`：Candidate Recall@20、Hit@5、MRR@5、引用有效性、任务完成率及 trace 成本/耗时/行为指标
- [x] T13.4a `eval/retrieval_runner.py`：共享索引运行 R1—R4，输出候选 / Top5 / gold 结构化 raw
- [x] T13.4b `eval/orchestration_runner.py`：固定 Planner、query cache、P1/P2 与 Q1/Q2 编排对照
- [x] T13.5a `eval/report.py`：从 R 轨 raw 重算指标并生成 Markdown 对照表
- [x] T13.5b 报告生成器接入 P/Q raw + trace 汇总与组间差值
- [x] T13.6a 跑完 100 题 R1—R4，产出 400 条真实检索观测
- [ ] T13.6b 跑完 P1/P2 并行微基准与 Q1/Q2 Critic 对照，产出真实编排数据
- [x] T13.7a 把 R 轨关键数字回填到 `README.md` 与 `RESUME_MAPPING.md`
- [ ] T13.7b P/Q 轨完成后回填耗时、成本、覆盖与引用数字

**验收标准**

- 四组实验数据完整，每组指标可复现
- 每一项优化（混合检索 / 重排 / 并行 / 反思）都有独立归因的数字
- **红线**：报告中不出现任何未实际跑出的估计值

---

## Phase 14: MCP Server

- [ ] T14.1 `mcp/server.py`：FastMCP 暴露 `deep_research` 与 `kb_search`
- [ ] T14.2 工具描述与参数 schema 编写（面向 LLM 客户端的可读性）
- [ ] T14.3 在 Claude Code 中实测调用成功
- [ ] T14.4 README 补充客户端配置片段与调用截图
- [ ] T14.5 `tests/test_mcp_server.py`

**验收标准**

- 在真实 MCP 客户端中能看到工具、成功调用并拿到结果
- 有可直接复制的配置示例

---

## Phase 15: 工程化

- [ ] T15.1 `Dockerfile.backend`（多阶段构建，控制镜像体积）
- [ ] T15.2 `Dockerfile.frontend`
- [ ] T15.3 `docker-compose.yml`（含向量库与 trace 目录挂载）
- [ ] T15.4 干净环境验证 `docker-compose up`
- [ ] T15.5 `.github/workflows/ci.yml`：ruff lint + pytest
- [ ] T15.6 CI 中排除需要真实 API Key 的测试（标记 `@pytest.mark.live`）

**验收标准**

- 干净环境一键启动并完成一次研究任务
- CI 徽章为绿色，且不依赖任何私密 Key

---

## Phase 16: 交付物

> 这一 Phase 决定项目在简历场景下的实际价值，优先级不低于任何技术 Phase。

- [x] T16.1 重写 `README.md`：定位一句话、架构图、快速开始、评测对照表、技术决策摘要（真实指标待 Phase 13 回填）
- [x] T16.2 架构图（ASCII 或图片二选一，保证 GitHub 上直接可见）
- [ ] T16.3 Demo 截图 / GIF
- [ ] T16.4 `RESUME_MAPPING.md` 回填真实数字
- [ ] T16.5 中文技术口述稿（每个模块 3 分钟讲清"做了什么 / 为什么这么做 / 数据是多少"）
- [ ] T16.6 推送 GitHub，仓库名 `deepresearch-agent`，补充 topics 与简介

**验收标准**

- README 在不看代码的前提下能让人明白系统做什么、怎么做、效果如何
- 每条简历 bullet 都能在仓库中指到具体文件
- 口述稿覆盖 `RESUME_MAPPING.md` 中列出的全部面试问题

---

## 执行约定

沿用 v1 的阶段检查点，每个 Phase 完成后：

1. 该 Phase 全部任务勾选
2. `uv run pytest` 全绿（含 v1 原有 41 条，**回归红线**）
3. 能用中文讲清本阶段做了什么、为什么这么做
4. git commit，message 标注 Phase 编号

**跨 Phase 红线**：任何一个 Phase 都不得破坏 v1 已通过验收的功能。若必须破坏，先在本文档记录原因与迁移方案。
