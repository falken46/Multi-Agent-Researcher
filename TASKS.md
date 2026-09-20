# DeepResearch Agent - 任务拆解清单 (v2 / v3)

> Phase 0—9 为 v1，已全部完成（详见 git 历史）。
> v2 为 Phase 10—16，已完成。
> **v3 从 Phase 17 开始，变更说明见 `UPGRADE_V3.md`。**
> 每个 Phase 完成后暂停验收，并单独 commit。

---

## 进度总览

| Phase | 内容 | 里程碑 | 状态 |
|-------|------|--------|------|
| 10 | 基础设施层（config / llm / trace / costs） | M1 | ✅ 已完成 |
| 11 | RAG 检索层 | M1 | ✅ 已完成 |
| 12 | Agent 编排升级（Critic / 并行 / Checkpoint） | M1 | ✅ 功能完成 |
| 13 | 评测体系 | M2 | ✅ 秋招功能范围已完成 |
| 14 | MCP Server | M3 | ✅ 功能完成 |
| 15 | 工程化（Docker / CI） | M2 | ✅ 功能完成 |
| 16 | 交付物（README / 架构图 / 简历映射 / 口述稿） | M1—M3 | 🟨 部分完成 |
| 17 | 向量库迁移 Chroma → Milvus | M4 | ✅ 已完成 |
| 18 | 知识库语料替换与评测重跑 | ~~M4~~ **P2** | ⏸ 降级为可选（见 `UPGRADE_V3.md` §2.1） |
| 19 | 关系数据库层（PostgreSQL / SQLAlchemy / Alembic） | M4 | ✅ 已完成 |
| 20 | 可观测 exporter + BM25 增量 + 接口鉴权 | M5 | 🟨 代码全部完成；仅缺 T20.3 看板截图（需真实后端） |
| 21 | fork 基线冻结与交付物更新 | M4—M5 | 🟨 文档已更新（改名/推送待作者） |

**里程碑定义**

- **M1 可投最低线**：Phase 10—12 + Phase 16 的 README 部分完成 → GitHub 上线，简历可挂链接
- **M2 完整线**：+ Phase 13、15 → 有评测数据与工程化证据
- **M3 加分线**：+ Phase 14 → MCP Server 与 Demo 录制
- **M4 技术栈对口线（v3）**：+ Phase 17、19、21 → 企业档向量库 + 关系数据库落库（Phase 18 换语料已降为 P2 可选）
- **M5 工程完备线（v3）**：+ Phase 20 → 集中式可观测、增量索引、接口鉴权

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
- [x] 并发边界由 fake IO + trace latency 确定性测试验证；真实耗时对照转为秋招后可选实验，不作为功能验收或对外指标
- [x] 中断任务后可从 Checkpoint 恢复

> ⭐ **Phase 12 完成即达 M1 可投最低线**，此时应立即执行 Phase 16 的 README 部分并推送 GitHub，不要等后续 Phase。

---

## Phase 13: 评测体系

> 目标：用公开数据产出可复现的检索数字，并完成端到端评测基础设施。P/Q 真实付费运行根据秋招时间收益比转为可选，不阻塞后续功能阶段。

- [x] T13.1 接入公开 `C-MTEB/T2Reranking` 检索集，并构造 15 题端到端编排集
- [x] T13.2 用公开 positive / hard negative / qrels 生成共享 passage 池，替代人工 gold 标注
- [x] T13.3 `eval/metrics.py`：Candidate Recall@20、Hit@5、MRR@5、引用有效性、任务完成率及 trace 成本/耗时/行为指标
- [x] T13.3b 补多正例排序指标：Recall@5、nDCG@5、MAP@20（见下方「指标口径修订」）
- [x] T13.4a `eval/retrieval_runner.py`：共享索引运行 R1—R4，输出候选 / Top5 / gold 结构化 raw
- [x] T13.4b `eval/orchestration_runner.py`：固定 Planner、query cache、P1/P2 与 Q1/Q2 编排对照
- [x] T13.4c runner 在候选深度重排并落 `ranked_chunk_ids`（schema_version 2），使深层指标可算
- [x] T13.5a `eval/report.py`：从 R 轨 raw 重算指标并生成 Markdown 对照表
- [x] T13.5b 报告生成器接入 P/Q raw + trace 汇总与组间差值
- [x] T13.5c 报告对缺 `ranked_chunk_ids` 的旧 raw fail-fast，杜绝浅层数字冒充 MAP@20
- [x] T13.6a 跑完 100 题 R1—R4，产出 400 条真实检索观测
- [x] T13.7a 把 R 轨关键数字回填到 `README.md` 与 `RESUME_MAPPING.md`

### 秋招后可选实验

- P1/P2 并行微基准与 Q1/Q2 Critic 对照的真实付费运行
- 只在正式 P/Q raw 生成后回填耗时、成本、覆盖与引用数字；未运行前不保留占位数字

### 指标口径修订（2026-08-26）

首版 R 轨只有 Candidate Recall@20 / Hit@5 / MRR@5，三者都是「任一命中」或「首个命中」型指标。
但 `eval/dataset/t2_reranking/metadata.json` 显示 100 个 query 共 755 条 positive 关系，
**平均每 query 7.55 个正例**：此时 Hit@5 在 93—96% 已接近饱和，MRR@5 只反映第一个正例的位置，
而 reranker 的实际工作是把全部正例顶到负例之上——原有指标看不见这件事。
T2Reranking 在 C-MTEB 的官方主指标本就是 MAP，正因为它是多正例数据集。

因此追加 Recall@5、nDCG@5、MAP@20 三项，并据此重判 rerank 结论。

**验收标准**

- R1—R4 四组公开检索实验数据完整，每组指标可由结构化 raw 复现
- P/Q runner、query cache、断点续跑与组间报告已通过离线功能测试
- 深层指标必须来自含 `ranked_chunk_ids` 的 raw；旧格式一律 fail-fast
- **红线**：报告中不出现任何未实际跑出的估计值

---

## Phase 14: MCP Server

- [x] T14.1 `mcp_server/server.py`：官方 MCP Python SDK v2 `MCPServer` 暴露 `deep_research` 与 `kb_search`
- [x] T14.2 工具描述、参数/输出 JSON Schema 与行为 annotations（面向 LLM 客户端）
- [x] T14.3 官方 SDK 客户端通过真实 stdio 子进程完成握手、工具发现与 `kb_search` 结构化调用
- [x] T14.4 README 与 `.mcp.json` 补充可复制的客户端配置；截图统一留到 T16.3 Demo
- [x] T14.5 `tests/test_mcp_server.py`

**验收标准**

- 官方 MCP 客户端能通过真实 stdio 子进程看到工具、调用 `kb_search` 并拿到结构化结果
- Claude Code 已识别项目级配置；实际工具结果发送给外部模型属于可选的数据出站验收，不阻塞本地协议功能完成
- 有可直接复制的配置示例

---

## Phase 15: 工程化

- [x] T15.1 `Dockerfile.backend`（多阶段构建、非 root 运行、健康检查）
- [x] T15.2 `Dockerfile.frontend`（多阶段构建、非 root 运行、健康检查）
- [x] T15.3 `docker-compose.yml`（一次性 indexer + 向量库 / BM25 / checkpoint / trace / 模型缓存卷）
- [x] T15.4 干净 Docker 卷验证启动：44 篇文档 → 128 chunk，两路索引各 128 条，后端与前端均健康
- [x] T15.5 `.github/workflows/ci.yml`：锁定依赖 + Ruff + pytest，第三方 Action 固定完整 commit SHA
- [x] T15.6 注册 `@pytest.mark.live`，CI 显式执行 `pytest -m "not live"` 且不注入任何 API Key

**验收标准**

- 标准环境可用 `docker compose up --build` 自动建库并启动前后端；本机已用干净命名卷验证完整启动链和健康检查
- 真实 DeepSeek / Web 研究任务按作者的功能优先决策保留为可选付费 smoke，不把未运行结果写成验收数字
- CI 本地等价命令已通过且不依赖任何私密 Key；GitHub 徽章需在作者推送后由远端 workflow 生成

---

## Phase 16: 交付物

> 这一 Phase 决定项目在简历场景下的实际价值，优先级不低于任何技术 Phase。

- [x] T16.1 重写 `README.md`：定位一句话、架构图、快速开始、真实 R 轨对照表、交付证据与技术决策摘要
- [x] T16.2 架构图（ASCII 或图片二选一，保证 GitHub 上直接可见）
- [ ] T16.3 Demo 截图 / GIF
- [x] T16.4 `RESUME_MAPPING.md` 回填 R 轨、MCP、离线回归与 Docker 本机验证证据，并明确 P/Q 和远端 CI 边界
- [x] T16.5 `INTERVIEW_GUIDE.md` 中文技术口述稿（编排、检索、评测、可观测、MCP、Docker / CI）
- [ ] T16.6 推送 GitHub，仓库名 `deepresearch-agent`，补充 topics 与简介（`RELEASE_CHECKLIST.md` 已备好；待作者自行提交、改名与推送）

**验收标准**

- README 在不看代码的前提下能让人明白系统做什么、怎么做、效果如何
- 每条简历 bullet 都能在仓库中指到具体文件
- 口述稿覆盖 `RESUME_MAPPING.md` 中列出的全部面试问题

---

# v3（Phase 17 起）

> 变更说明与决策留档见 `UPGRADE_V3.md`。v3 是**技术栈升级，不增加研究能力**。
> Phase 17—19 产出的基础设施同时是金融版（合规审查 Agent）的 fork 基线，只做一次。

## Phase 17: 向量库迁移 Chroma → Milvus

> 目标：把向量存储从原型档换到企业档，同时**验证 v2 当初留下的窄接口设计**。
> **本 Phase 不换语料** —— 控制变量，见 `UPGRADE_V3.md` §4.2。

- [x] T17.0 记录基线：`pytest --collect-only` 数量、全量通过情况，填入 `UPGRADE_V3.md` §7
- [x] T17.1 `docker-compose.yml` 增加 milvus + etcd + minio 三个服务，配 healthcheck（`docker compose config` 校验通过；**完整启动链未实跑**）
- [x] T17.2 `rag/vectorstore.py` 新增 `MilvusVectorStore`，方法签名与 `ChromaVectorStore` 完全一致（`add` / `query` / `count`）
- [x] T17.3 `core/config.py`：`chroma_dir` / `chroma_collection` → `milvus_uri` / `milvus_collection`，同步更新 `.env.example` 与 `TECH_STACK.md`
- [x] T17.4 `rag/pipeline.py:74`、`:157` 两处实例化改用新实现
- [x] T17.5 `eval/retrieval_runner.py` 五处引用适配
- [x] T17.6 测试适配：`tests/test_core_config.py`、`tests/test_rag_pipeline.py`；补 Milvus collection/schema/index 相关用例
- [x] T17.7 提供 Milvus Lite（或 fake）替身，保证 CI 不起完整 Milvus 仍能跑
- [x] T17.8 用**原语料**重建索引，跑一遍 R 轨（R1/R2/R3 实跑，18 项指标与 Chroma 记录一致；R4 未跑，仅重排同一候选集）

**验收标准**

- **`rag/hybrid.py` 与 `rag/rerank.py` 一行未改**，其现有测试全绿 ← 本 Phase 最重要的验收信号
- T17.8 的 R 轨结果与 v2 Chroma 版**基本一致**；不一致先查适配器，不得直接进入 Phase 18
- 离线可跑不变：`EMBEDDING_BACKEND=fake` 时不下载模型、不发网络请求
- 基线测试数量的增减去向说得清

---

## Phase 18: 知识库语料替换与评测重跑（⏸ P2 可选）

> **2026-09-08 降级为可选。** 初稿把它排成 M4 必做，依据是"语料与查询同源会削弱评测可信度"——
> 该依据经核实**不成立**：R 轨跑的是 C-MTEB 公开基准，与 `data/kb` 无关。
> 换语料只有观感收益、无技术后果，且保留自述文档有"零准备可复现 + 答案可验证"的反面论证。
> 完整论证见 `UPGRADE_V3.md` §2.1。
>
> 若将来要做：**只能换中文语料**（embedding 与 reranker 都是中文模型），
> 且需同步改 `tests/test_rag_pipeline.py:32` 的 `13_rrf.md` 断言与多处文档示例。

- [ ] T18.1 确定语料方向并落地下载（🔶 **阻塞决策**，见 `UPGRADE_V3.md` §8）
- [ ] T18.2 建立来源清单：每篇记录标题、来源 URL、获取日期
- [ ] T18.3 归档旧 `data/kb/` 44 篇（**不删**，作为 v2 评测的可复现材料保留）
- [ ] T18.4 重建向量索引与 BM25 索引，记录文档数与 chunk 数
- [ ] T18.5 R 轨标注集适配新语料
- [ ] T18.6 重跑四组消融，产出与 `eval/reports/comparison.md` 同格式的新对照表
- [ ] T18.7 撰写新旧结论的异同说明：哪些结论迁移过来仍成立，哪些变了，为什么

**验收标准**

- 新语料的四组对照表产出，**负向结果照实保留**
- v2 的 C-MTEB 结论与指标口径讨论**原样保留**，与新结论并列呈现，讲成"公开基准验证方法 → 垂直语料验证迁移"两段
- README 里不再出现"知识库是本项目自己的文档"这一事实

---

## Phase 19: 关系数据库层

> 目标：补齐全项目零业务数据持久化这个空洞。
> 本 Phase 产出的 SQLAlchemy / Alembic / session / repository 骨架，是金融版直接继承的部分。

- [x] T19.1 `docker-compose.yml` 增加 postgres 服务 + healthcheck，backend 加 `depends_on: service_healthy`
- [x] T19.2 `db/models.py`：`research_tasks` / `reports` / `citations` / `sub_questions` 四张表（列定义见 `UPGRADE_V3.md` §4.3）
- [x] T19.3 `db/migrations/`：Alembic 初始化 + 首个迁移
- [x] T19.4 `db/repository.py`：固定模板的数据访问层，**模型不生成 SQL**
- [x] T19.5 图运行结束后落库：任务、报告、子问题、引用（含 `retrieval_score` 与 `rank`）
- [x] T19.6 db 层测试：CRUD、级联、事务回滚、`alembic upgrade head` 从空库建全表
- [x] T19.7 CI 增加 PostgreSQL service container

**验收标准**

- 跑一次 `/research` 后，能从数据库查到完整链路：问题 → 子问题 → 引用来源及其检索分数与名次 → 报告
- `alembic upgrade head` / `downgrade` 双向可用
- 未配置 `DATABASE_URL` 时，系统仍能运行（落库降级为跳过并告警），保住离线可跑

---

## Phase 20: 可观测 exporter + BM25 增量 + 接口鉴权

> 目标：工程完备度补齐。三项互相独立，可分开验收、可按时间砍。

- [x] T20.1 接 **OpenTelemetry** exporter（`core/otel.py`），事件模型一行未改；Laminar / Langfuse 均可作为后端
- [x] T20.2 本地 JSONL 保留为主路径，未配置 `OTEL_ENDPOINT` 时完全离线可跑
- [ ] T20.3 trace 视图截图放进 README（**需作者注册 Laminar/Langfuse 云端免费账号跑一次**；本地无服务无法产出）
- [x] T20.4 `rag/bm25.py` 支持增量追加，不再全量覆盖重建
- [x] T20.5 `backend/api.py` 扩到任务生命周期接口（发起 / 查询 / 流式 / 取报告）；任务开始落 `running`、正常完成落 `completed`、异常落 `failed`，恢复同一 `thread_id` 更新原记录
- [x] T20.6 API Key 鉴权，调用主体写入 trace 事件与 `research_tasks`

**验收标准**

- 换 exporter 后业务代码零改动
- 增量追加一篇文档后，索引条数正确且原有条目未丢失
- 无 API Key 请求返回 401

---

## Phase 21: fork 基线冻结与交付物更新

> 目标：把 Phase 17—19 的成果冻结成金融版的分叉点，并更新全部对外材料。

- [ ] T21.1 Phase 19 完成后打 tag（建议 `v3-fork-base`），作为合规审查 Agent 的分叉基线
- [ ] T21.2 分叉出合规版仓库，其 README 写明 fork 来源与迁移范围（**血缘声明，见 `UPGRADE_V3.md` §3.1**）
- [x] T21.3 更新本项目 `README.md`：Milvus、新语料、数据库、新旧评测对照
- [x] T21.4 更新 `ARCHITECTURE.md`：向量库与持久化两处
- [x] T21.5 更新 `TECH_STACK.md` v3 依赖变更
- [x] T21.6 更新 `RESUME_MAPPING.md`：新增 bullet 需指到文件与数据来源；**不得与合规版重复声明同一份工作量**
- [x] T21.7 更新 `INTERVIEW_GUIDE.md`：补 Milvus 选型、控制变量迁移顺序、数据表设计、两个项目血缘关系的答法
- [ ] T21.8 仓库改名 `deepresearch-agent`（`RELEASE_CHECKLIST.md` 已备好）

**验收标准**

- 两个仓库的 README 都能说清彼此关系，且无重复计工作量
- `UPGRADE_V3.md` §7 基线表填完，测试增减去向有交代
- 面试口述稿能回答："这两个项目是什么关系？"

---

## 执行约定

沿用 v1 的阶段检查点，每个 Phase 完成后：

1. 该 Phase 全部任务勾选
2. `uv run pytest` 全绿（含 v1 原有 41 条，**回归红线**）
3. 能用中文讲清本阶段做了什么、为什么这么做
4. git commit，message 标注 Phase 编号

**跨 Phase 红线**：任何一个 Phase 都不得破坏 v1 已通过验收的功能。若必须破坏，先在本文档记录原因与迁移方案。
