# DeepResearch Agent — v3 升级方案

> 版本：v3.0（规划中）
> 上一版：v2.0（Phase 10—16，已完成）
> 本文是 v3 的**变更留档**，任务拆解在 `TASKS.md` Phase 17—21，依赖变更在 `TECH_STACK.md` §2.4。
> 创建日期：2026-09-08

---

## 1. 版本说明：v2 → v3 变更概览

v2 交付了一个能力完整的深度研究系统。v3 **不增加新的研究能力**，只做两件事：把技术栈从"原型档"抬到"企业档"，以及修掉一个一直没处理的硬伤。

| 维度 | v2 现状 | v3 目标 |
|------|---------|---------|
| 向量库 | Chroma（嵌入式、零运维、原型档） | **Milvus**（独立服务、企业档，国内企业侧主流） |
| 知识库语料 | `data/kb/` 44 篇本项目自己的架构说明 | ~~真实外部语料~~ **（P2 可选，见 §2.1）** |
| 业务数据持久化 | **无**。`/research` 跑完只留 trace 文件，任务与报告都不落库 | **PostgreSQL**：研究任务、报告、引用来源落库 |
| BM25 索引 | pickle 整包覆盖重建，加一篇文档要重建全量 | 支持**增量追加** |
| 可观测落地 | 本地 JSONL + 进程内锁，多副本会散 | **OpenTelemetry 导出**（后端可为 Laminar / Langfuse 等任意 OTLP 接收端）；**JSONL 仍是主路径**，不降级 |
| 对外接口 | `/health` + `/research` 两个，无鉴权 | 任务生命周期接口 + API Key |

---

## 2. 为什么做这次升级

### 2.1 观感问题：知识库装的是项目自己的文档（**已降级为可选，2026-09-08**）

`data/kb/` 下 44 篇全部是 `01_langgraph_stategraph.md`、`13_rrf.md` 这类**本项目的架构说明**。
一个"深度研究系统"检索自己的设计文档，读者第一眼会觉得循环、像玩具。

> ⚠️ **本节初稿有一处事实错误，已更正**：初稿写「它会削弱评测结论的可信度（语料与查询同源）」，
> **这是错的**。R 轨评测跑的是 `eval/dataset/t2_reranking`（C-MTEB/T2Reranking 公开基准），
> 与 `data/kb` 完全无关。`docs/study/04_实验与结论.md:136` 本来就写明了两者职责不同：
> `data/kb/` 是产品演示知识库，公开基准才是评测数据。**换语料不影响任何评测数字。**

因此这一项**只有观感收益，没有技术后果**，且存在不弱的反面论证：

| 保留自述文档的理由 | 说明 |
|---|---|
| 零准备可复现 | clone 下来直接 `index_cli --dir data/kb` 就能跑，不用找语料、不涉版权 |
| 答案可验证 | 读者能打开 `13_rrf.md` 核对系统答得对不对；换成外部论文则无从判断答案是否编造 |

**可辩护的对外说法**：「演示语料用项目自己的技术文档，是为了任何人零准备就能复现，
且能对着源文档验证答案；评测走 C-MTEB 公开基准，两者职责分开。」

另外，初稿建议的 **arXiv 论文方案不可行**：`EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5` 与
`RERANK_MODEL=BAAI/bge-reranker-base` 都是中文模型，换英文语料会掉检索质量；
换 embedding 模型则整个向量库作废、评测前后不可比（见 §5 与 `TECH_STACK.md` §8）。
**真要换只能换中文语料。**

换语料的实际成本：`tests/test_rag_pipeline.py:32` 的断言写死了 `13_rrf.md`；
`README.md` / `ARCHITECTURE.md` / `docs/study/` 多处示例要同步改。

**结论：Phase 18 从 M4 必做降为 P2 可选**（作者 2026-09-08 决定）。
优先级低于 Phase 19 —— 后者补的是 JD 里 35% 的真实技能空洞，收益是硬的。

### 2.2 技术栈档次：Chroma 与"零业务数据持久化"

参照 2026-07 一份 60 条真实 JD 的人工编码调研（中国大陆样本 20 条）：

| JD 技术方向 | 中国大陆出现率 | v2 现状 |
|---|---|---|
| 向量库 / Embedding / 搜索 | 70% | ⚠️ Chroma 属原型档，企业侧主流是 Milvus / pgvector |
| SQL / 关系数据库 / 数据工程 | 35% | ❌ **完全空白** |
| 后端 / API / 微服务 | 75% | ⚠️ 只有 2 个接口 |
| 评测 / 监控 / LLMOps | 60% | ✅ 已有，但落地方式是本地文件 |

> 口径提醒：人工编码的 20 条样本，一条 JD = 5 个百分点，相差 5—10% 不应解读为高低差。趋势参考，不是官方统计。

### 2.3 这批基础设施是与金融版共享的 fork 基线

见下节。**同一份基础设施做一次，两个项目都受益**，这是把它放在本项目主干而不是分支上做的原因。

---

## 3. 与合规审查版（金融版）的关系

合规审查 Agent（金融版，规划文档在 `实习材料/证券合规审查Agent/`）是一个独立项目，代码源自本项目。

**⚠️ 分叉时点已于 2026-09-08 由作者调整**：原计划等本项目 Phase 19 完成后再 fork；
现改为**直接从当时的代码基线开始**，Milvus 迁移、规则语料与 PostgreSQL 基础设施
各自在合规版内独立完成，不再把本项目的 v3 进度当作前置条件。

```
DeepResearch Agent 主干
  ├─ Phase 17  Chroma → Milvus          ✅ 已完成（2026-09-08）
  ├─ Phase 18  知识库换真实语料          ⏸ 降为 P2 可选（2026-09-08）
  ├─ Phase 19  PostgreSQL 基础设施
  ├─ Phase 20  可观测 exporter + BM25 增量
  └─ Phase 21  交付物更新

合规审查 Agent（独立维护）
  └─ 从 2026-09-08 的代码基线导入，自行推进
     （领域语料 / 8 张业务表 / HITL / 审计闭环 / fail-closed / 私有化部署）
```

> Phase 17 已于 2026-09-08 完成，因此**合规版导入的基线里已经带着 Milvus 适配器与那三个坑的修复**
> （见 §7.4）—— 这部分不需要在合规版重做一遍。
>
> Phase 21 的原「fork 基线冻结」任务因此失去前置意义，保留为交付物更新即可。

### 3.1 血缘声明（红线）

两个项目共享同一个祖先，这是事实，**GitHub 的 commit 历史、代码结构与相似度都藏不住**。

因此：

- ✅ **必须**在合规版的 README 里写明它 fork 自本项目，并说明迁移了哪些部分（检索层、评测框架、LLM 网关、编排骨架）
- ✅ 对外讲法：「第二个项目是在第一个的基础上重做的，检索层和评测方法整体迁移，然后针对合规场景重新设计了落库、人工复核与失效处置」—— 这讲的是**方法沉淀与复用能力**
- ❌ **不许**把两者表述为"两个从头独立开发的项目"
- ❌ **不许**在两个仓库里重复声明同一份工作量（例如两边都写"148 项测试"）

危险的从来不是"存在两个项目"，是"假装它们无关"。主动说出血缘，反而是工程能力的证明。

---

## 4. 逐项改造说明

### 4.1 Chroma → Milvus（Phase 17）

**现状（已核实）**：`ChromaVectorStore` 只暴露三个方法 `add()` / `query()` / `count()`，适配器共 113 行。

全项目引用点：

| 位置 | 用途 |
|---|---|
| `rag/pipeline.py:74`、`rag/pipeline.py:157` | 两处实例化 |
| `eval/retrieval_runner.py:36/101/151/153/178` | 评测轨使用 |
| `core/config.py:68-69` | `chroma_dir` / `chroma_collection` |
| `tests/test_core_config.py:16`、`tests/test_rag_pipeline.py:90-91` | 两处测试 |

**`rag/hybrid.py`（RRF）与 `rag/rerank.py` 完全不引用它** —— 当初的窄接口设计在这里得到验证。`data/kb/34_vector_index_operations.md` 里就写过这个意图：

> 窄接口减少业务代码对 Chroma API 的耦合，未来更换向量库时，pipeline 不需要理解供应商细节。

**目标**：新增 `MilvusVectorStore`，方法签名与 `ChromaVectorStore` 完全一致；`pipeline.py` 换实例化；配置字段换成 `milvus_uri` / `milvus_collection`；compose 增加 milvus + etcd + minio。

**验收信号（关键）**：`rag/hybrid.py` 与 `rag/rerank.py` **一行不改**，其现有测试全绿。这检验的不是"会不会用 Milvus"，是**当初接口有没有设计干净**。

**诚实的代价**：本项目语料规模（几十篇文档、百余 chunk）**Chroma 完全够用**。换 Milvus 是为技术栈对口，不是性能需要，代价是多三个容器和一份运维复杂度。对外表述必须带上这句，不要编造性能理由。

### 4.2 知识库语料替换（Phase 18）

**现状**：`data/kb/` 44 篇自述文档，构建为 128 个 chunk。

**⏸ 本节已降级为 P2 可选，论证见 §2.1。** 以下保留为将来真要做时的参考。

**目标**：换成真实外部语料。**只能选中文语料** —— embedding 与 reranker 都是中文模型
（`bge-small-zh-v1.5` / `bge-reranker-base`），换英文会掉检索质量；换模型则向量库作废、评测不可比。

~~arXiv 公开论文~~ ← 初稿建议，因语言不匹配已否决。

**待作者决策**：具体中文语料方向。选定后需记录来源清单（标题、来源 URL、获取日期），
并同步改 `tests/test_rag_pipeline.py:32` 的 `13_rrf.md` 断言。

**⚠️ 顺序上的关键设计**：**先做 Phase 17（换向量库、保持旧语料），再做 Phase 18（换语料）。**

理由是控制变量：

| Phase | 变量 | 期望结果 | 说明 |
|---|---|---|---|
| 17 | 只换向量库 | R 轨结果与 Chroma **基本一致** | 一致才说明迁移正确；不一致说明适配器有 bug |
| 18 | 只换语料 | R 轨结果**会变** | 这次的差异才是语料带来的，可以解释 |

两个一起换，出了差异就分不清是迁移错了还是语料变了。这与本项目在 C-MTEB 上做四组消融是同一套控制变量思路。

### 4.3 关系数据库层（Phase 19）

**现状**：全项目零业务数据持久化。`checkpoints.sqlite` 存的是 LangGraph 状态，chroma/bm25 存的是索引，都是运行时产物，不是业务数据。一次 `/research` 跑完，任务和报告都不落库。

**目标**：PostgreSQL + SQLAlchemy + Alembic。最小表集：

| 表 | 关键列 | 用途 |
|---|---|---|
| `research_tasks` | `id, question, thread_id, status, model_name, token_in, token_out, cost, started_at, finished_at` | 一次研究任务 |
| `reports` | `id, task_id, content, quality_score, revision_count, created_at` | 产出的报告 |
| `citations` | `id, report_id, chunk_id, source_type, source_ref, retrieval_score, rank` | 引用来源，**存分数与名次** |
| `sub_questions` | `id, task_id, seq, question, status, fallback_triggered` | Planner 拆出的子问题 |

**两条设计约束**（与合规版共享同一套思路）：

1. `citations` 必须存 `retrieval_score` 与 `rank`，不能只存关联 —— 否则事后无法回答"当时为什么是这条排第一"
2. 模型永远不生成 SQL，只输出结构化字段，SQL 由服务端按固定模板装配

**为什么加 Alembic 而不只是建表**：`CREATE TABLE IF NOT EXISTS` 只能建新表、改不了已有表。有迁移脚本才算完整的数据库工程。

### 4.4 BM25 增量更新（Phase 20）

**现状**：`rag/bm25.py:28` 的 `add()` 是 `self._chunks = list(chunks)` 全量覆盖 + 重建 engine + 整包 `pickle.dump`。加一篇文档要重建整个索引，且索引全量常驻内存。

**目标**：支持增量追加，不必全量重建。

**不做的**：换 Elasticsearch。收益明确（增量 + 企业检索事实标准），但工作量大，排在 v3 范围外。

### 4.5 可观测 exporter（Phase 20）

**现状**：`core/trace.py:55` 写 `traces/日期/trace_id.jsonl`，用 `threading.Lock` 保证同进程内不串。多副本部署会各写各的，无法集中查询。

**目标（已完成）**：接 **OpenTelemetry**，而不是绑定某一家的 SDK。事件模型
（`task_start` / `task_end` / `fallback` / revision 事件 + token 成本）**一行未改**，
只在落地层叠加一个 `core/otel.py`。

**为什么是 OTel 而不是 Langfuse SDK**：Laminar 是 OTel 原生，Langfuse 也接受 OTLP，
所以真正的问题不是"选哪家"，而是"绑标准还是绑 SDK"。选标准之后换后端只改
`OTEL_ENDPOINT` 与 `OTEL_HEADERS` —— 与 `rag/vectorstore.py` 用窄接口隔离
向量库供应商是同一套思路。选型依据见 §7.4。

**事件流 → span 树的映射**：`task_start` / `task_resume` 开根 span，`node_start` / `node_end`
开关子 span，其余（`llm_call` / `retrieval` / `fallback` / `revision` / `error`）
挂成 span event。没有配对的事件不丢，也不会变成一堆零长度的孤立 span。

**本地 JSONL 仍是主路径，不是 fallback** —— 未配置 endpoint 时 OTel 层完全不介入。
这是 v2 "无 API Key 也能复现"这一优点的保命条款，不得移除。

### 4.6 接口与鉴权（Phase 20，P2）

`/health` + `/research` 扩到任务生命周期（发起 / 查询 / 流式 / 取报告），加 API Key 鉴权，调用主体写入 trace 事件与任务记录。

---

## 5. v3 明确不做

| 不做 | 理由 |
|---|---|
| 改动 `rag/hybrid.py`（RRF）与 `rag/rerank.py` | 它们是本项目的技术纵深所在，v3 的验收信号恰恰是"它们不用改" |
| 重跑 C-MTEB 四组消融的结论 | v2 的公开基准结论**原样保留**，它是方法论证据。v3 新增的是领域语料上的对照，两者并列呈现 |
| 换 embedding 模型 | 换模型等同于整个向量库失效，必须重建索引 + 重跑评测，评测数据前后不可比 |
| Elasticsearch | 见 §4.4 |
| 前端重写 / K8s / 微调 | 与 v2 的非目标一致 |
| 增加新的研究能力 | v3 是技术栈升级，不是功能升级 |

---

## 6. 回归红线

沿用 v2 `TASKS.md` 的跨 Phase 红线，并追加：

1. **动代码前先记基线**：`uv run pytest --collect-only -q` 与全量通过情况写入本文档 §7，每个 Phase 结束要能说清测试数量的增减去向。说不清 = 该 Phase 未完成
2. **v2 已通过验收的功能不得破坏**。若必须破坏，先在本文档记录原因与迁移方案
3. **Phase 17 的 R 轨结果必须与 Chroma 基本一致**，不一致先查适配器，不要急着换语料
4. **`eval/reports/comparison.md` 与 `EVAL.md` §3.1 的指标口径讨论不得删除**，它们是 v2 最有说服力的部分
5. **远端 CI 绿色状态必须实际推送并看到绿灯之后才能声明**

---

## 7. 基线记录

### 7.1 Phase 17 开工前（2026-09-08）

| 项 | 数值 |
|---|---|
| `pytest --collect-only` 测试总数 | **148**（与 v2 文档记录精确吻合） |
| `pytest -m "not live"` | **148 passed in 6.69s** |
| `data/kb` 文档数 / chunk 数 | 44 篇 / 128 chunk |
| 向量索引条数 | 128 |

### 7.2 Phase 17 完成后（2026-09-08）

| 项 | 数值 | 增减去向 |
|---|---|---|
| 测试总数 | **162** | +14，全部来自新增 `tests/test_rag_vectorstore.py`；原 148 项**一项未删、一项未改** |
| `pytest -m "not live"` | **162 passed in 12.17s** | |
| `ruff check .` | All checks passed | |
| `rag/hybrid.py` / `rag/rerank.py` / `core/llm.py` | **0 处改动** | ← 本 Phase 的核心验收信号 |

### 7.3 Phase 19 完成后（2026-09-08）

| 项 | 数值 | 增减去向 |
|---|---|---|
| 测试总数 | **173** | +11，全部来自新增 `tests/test_db_repository.py` |
| `pytest -m "not live"` | **173 passed in 16.12s** | |
| `ruff check .` | All checks passed | |
| `rag/hybrid.py` / `rag/rerank.py` / `core/llm.py` / `agents/*` | **0 处改动** | 落库挂在 `backend/streaming.py` 收尾处，编排层未被侵入 |

新增：`db/models.py`（4 张表）、`db/session.py`、`db/repository.py`、`db/persistence.py`、
`db/migrations/`（Alembic + 首个迁移）、compose 的 `postgres` + `migrate` 服务、
CI 的 `database` job（真实 PostgreSQL）。

**Phase 19 的三个设计决策**

1. **落库是可选的（fail-open）**。`DATABASE_URL` 留空则整层静默跳过，
   写库失败只记日志不打断响应。保住 v2「无外部依赖也能跑完整离线测试」的优点。
   > 对照：合规审查版同一层是 **fail-closed** —— 写不进库就中止任务。
   > 同一段代码、两种场景、相反策略，这是有意的。
2. **模型方言中立**（不用 JSONB / ARRAY），所以本地与常规 CI 跑 SQLite、
   生产跑 PostgreSQL。但「在 SQLite 上过」不等于「在 PostgreSQL 上过」，
   因此 CI 单独加了一个 `database` job，用 `TEST_DATABASE_URL`
   把**同一套测试**对着真实 PostgreSQL 再跑一遍，并验证迁移在 PG 上双向可用。
3. **`citations` 必须存 `rank`**。只存"引用了这个来源"，事后就回答不了
   「当时为什么是这条排第一」——索引重建、模型升级后这个问题就永远没答案了。
   `retrieval_score` 上游暂未携带，**取不到就留空，不填 0**（0 会被误读成"相关度为零"）。

### 7.4 Phase 20 完成后（2026-09-09）

| 项 | 数值 | 增减去向 |
|---|---|---|
| 测试总数 | **208** | +35：OTel 12、BM25 增量 8、接口/鉴权/任务生命周期 15；原有测试一项未删 |
| `pytest -m "not live"` | **208 passed** | |
| `ruff check .` | All checks passed | |
| `rag/hybrid.py` / `rag/rerank.py` / `agents/*` | **0 处改动** | 三轮改造都没侵入编排与融合层 |

**Phase 20 的选型结论：用 OpenTelemetry，不绑定 Langfuse 或 Laminar。**

依据（2026-09-09 核实）：

| | Langfuse | Laminar |
|---|---|---|
| GitHub stars | 34.4k（2023-05 起） | 3.2k（2024-08 起，YC S24） |
| 归属 | **2026-01 起并入 ClickHouse** | 独立公司 |
| 与 OTel 的关系 | 集成 OTLP | **OTel 原生** |

两家都吃 OTLP，所以**真正的问题不是"选哪家"，而是"绑标准还是绑 SDK"**。
选 OTel 之后换后端只改 `OTEL_ENDPOINT` 与 `OTEL_HEADERS`，代码不动 ——
和 `rag/vectorstore.py` 用窄接口隔离向量库供应商是同一套思路。

**没有把任何一家塞进 compose**：两者自托管都要 ClickHouse + Postgres 等一整套，
而本项目 compose 已有 7 个服务，再加会到十几个 —— 一个 demo 项目摆十几个容器，
读者的第一反应不会是"厉害"，而是"在堆东西"。要看板截图用云端免费额度跑一次即可。

⚠️ **验证边界**：OTel 那 12 项测试用官方 `InMemorySpanExporter` 走真实 SDK 读回 span，
验证的是 span 树建对了；**没有验证真实 Laminar / Langfuse 能否收下这些 span**，
看板截图（T20.3）也尚未产出。这两条不能声称已完成。

### 7.5 Phase 17 的 A/B 对照结果（T17.8）

分两步验证，先小后大。

#### 第一步：知识库语料的排序对照

同一份 `data/kb`（44 篇 / 128 chunk），两个后端各自建索引后跑 6 条查询，
仅开向量通道、关闭重排以隔离变量：

| 项 | 结果 |
|---|---|
| 两边索引规模 | 均为 docs=44 / chunks=128 / vectors=128 |
| 排序完全一致 | **6/6 全部一致** |
| 最大分数偏差 | **0.00000030**（float32 舍入级别） |

样例（query =「RRF 为什么适合混合检索」）：两个后端 Top3 完全相同 ——
`13_rrf.md` (0.801) → `35_bm25_lexical_ranking.md` (0.6345) → `33_embedding_backend_design.md` (0.5975)。

#### 第二步：100 题 R 轨实跑对照（决定性证据）

`VECTOR_BACKEND=milvus` 重跑 C-MTEB/T2Reranking 的 R1/R2/R3（1664 passage / 100 query），
与 `eval/reports/comparison.md` 记录的 Chroma 版数字逐格比：

| 组 | CandRec@20 | Hit@5 | MRR@5 | Rec@5 | nDCG@5 | MAP@20 |
|---|---|---|---|---|---|---|
| R1 chroma（记录） | 0.9364 | 0.9600 | 0.7238 | 0.4546 | 0.6254 | 0.6204 |
| R1 milvus（实跑） | 0.9364 | 0.9600 | 0.7238 | 0.4546 | 0.6254 | 0.6204 |
| R2 chroma（记录） | 0.8565 | 0.9400 | 0.7575 | 0.4081 | 0.5961 | 0.5538 |
| R2 milvus（实跑） | 0.8565 | 0.9400 | 0.7575 | 0.4081 | 0.5961 | 0.5538 |
| R3 chroma（记录） | 0.9227 | 0.9500 | 0.7777 | 0.4389 | 0.6359 | 0.6144 |
| R3 milvus（实跑） | 0.9227 | 0.9500 | 0.7777 | 0.4389 | 0.6359 | 0.6144 |

**18 项指标全部一致，最大偏差 0.000045** —— 而 `comparison.md` 只记到小数点后四位，
所以这个偏差就是记录精度本身，不是计算差异。

> R2（纯 BM25）不经过向量库，本应逐位相同，实测确实如此 —— 它在这里起对照组作用，
> 证明差异不是来自评测流程的随机性。
> R4 未跑：它只是对 R3 的同一候选集做重排，R3 一致即可推定；且需下载 cross-encoder 权重，
> 收益与成本不匹配。**未跑就是未跑，不写推定数字。**

**结论：迁移正确，可以进入 Phase 18。**

---

## 7.6 Phase 17 踩到的三个坑（写给面试与将来的自己）

### 坑一：Milvus 的 `distance` 在 COSINE 下是相似度，不是距离

| | 返回值含义 | 换算 |
|---|---|---|
| Chroma | 余弦**距离**，越小越相似 | `similarity = 1 - distance` |
| Milvus (`metric_type="COSINE"`) | **余弦相似度本身**，越大越相似 | `similarity = distance` |

照抄 Chroma 的 `1 - distance` **不会报任何错**，只会把排序整个倒过来 —— 最相关的排最后。
实测验证：相同向量 `distance=1.0`，正交向量 `distance=0.0`。

已用 `test_milvus_cosine_semantics_not_inverted` 锁死。

### 坑二：新连接里集合是 `released` 状态，search 前必须 `load_collection()`

报错：`MilvusException code=101 ... call load() before search/get/query`。

**为什么单元测试抓不到**：建集合的那个 client 实例是隐式加载的，所以「建完立刻查」永远能过。
只有当**建索引和查询分处两次运行**时才暴露 —— 而这恰恰是 pipeline 的真实用法
（`build_index` 和 `search` 各自构造一次 store）。

**这个 bug 是 A/B 对照抓出来的，不是单元测试。** 而且它的表现极具迷惑性：向量通道整个
静默降级成空结果，日志里只有一行 `kb vector search degraded`，检索照常返回（走 BM25），
不看排序对比根本发现不了。

修复：`_ensure_loaded()` 在 `query()` 与 `count()` 前调用，每实例只挡一次（`load_collection` 幂等）。
已用 `test_milvus_query_from_a_fresh_instance_after_build` 锁死 —— 该测试**刻意用两个实例**。

### 坑三：环境变量名 `MILVUS_URI` 与 pymilvus 自己的配置撞车

pymilvus 在 `pymilvus/settings.py:12` **import 阶段**就执行 `os.getenv("MILVUS_URI")`，
随后在 `orm/connections.py:95` 强制按 `http[s]://host:port` 解析。

于是只要把 `MILVUS_URI` 设成 Milvus Lite 的本地文件路径（正是测试与 CI 的用法），
`import pymilvus` 当场抛 `ConnectionConfigException: Illegal uri`——**报错点在 import，
离配置很远，第一眼完全看不出是自己的环境变量造成的**。

修复：项目侧的环境变量改名为 **`MILVUS_ENDPOINT`**（`core/config.py` 用
`validation_alias` 声明，并开 `populate_by_name` 保证按字段名构造仍可用）。
已用 `test_milvus_endpoint_env_var_avoids_pymilvus_collision` 锁死。

> 这条也提醒：**给配置起名时要避开依赖库自己占用的环境变量**。

---

> 三个坑合起来正好印证了 §4.2 的顺序设计：**先换库、不换语料**。如果两件事一起做，
> 出了差异就分不清是适配器 bug 还是语料变化，坑二这种「静默变空」几乎必然被误判成语料效应。

---

## 8. 待作者决策

| # | 决策 | 是否阻塞 | 建议 |
|---|---|---|---|
| 1 | **新知识库装什么语料** | **阻塞 Phase 18** | arXiv 公开论文，方向待定 |
| 2 | Langfuse vs OpenTelemetry | 阻塞 Phase 20 | Langfuse（自带 LLM 看板，可截图进 README） |
| 3 | fork 出合规版的具体时点 | 阻塞 Phase 21 | Phase 19 完成后 |
| 4 | 仓库名 `Multi-Agent-----` 改名 | 否 | `deepresearch-agent`（v2 `RELEASE_CHECKLIST.md` 已备好） |

**Phase 17（Milvus 迁移）不依赖任何决策，可立即开工。**
