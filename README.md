# DeepResearch Agent

基于 LangGraph 的工程化多智能体研究助手：本地混合检索不足时联网，Critic 定向补查，并以 SSE、SQLite Checkpoint 和任务级 JSONL trace 支撑可观测、可恢复的 Markdown 报告生成流程。

> 当前代码基线已实现 Phase 10—15 的秋招功能范围，包括公开检索评测、MCP Server、Docker Compose 与离线 CI。P/Q 真实付费运行已转为秋招后可选实验，本文只填写由结构化 raw 复算的公开检索数字，不提前填写质量提升率或真实任务加速比。

## 核心能力

- **四 Agent 状态机**：Planner 拆题，Researcher 检索与归纳，Critic 评分并指出缺口，Writer 汇总成报告。
- **本地优先的混合检索**：Milvus 向量召回与 BM25 关键词召回经 RRF 融合，再按配置执行 rerank；本地检索失败或最高分低于配置阈值时才联网补查。向量库后端由 `VECTOR_BACKEND` 切换（`milvus` / `chroma`），上层融合与重排逻辑不感知供应商。
- **受控并发与失败隔离**：Researcher 使用 `asyncio.gather`、`Semaphore`、超时和 `return_exceptions=True` 并发处理独立子问题。
- **有边界的质量返工**：Critic 只让 Researcher 补查明确缺口，并通过返工上限和分数停滞检测防止死循环。
- **可观测、可恢复**：JSONL trace 记录节点、token、估算成本、耗时、降级和返工事件；SQLite Checkpoint 通过稳定 `thread_id` 支持 API 从最近 checkpoint 恢复。
- **业务数据落库**：PostgreSQL 保存任务、子问题、报告与引用来源（含检索名次），任务按 `running → completed / failed` 更新并用 `thread_id` 保持恢复幂等；SQLAlchemy + Alembic 管理表结构与迁移，未配置 `DATABASE_URL` 时整层静默跳过。
- **OpenTelemetry 导出**：事件流映射为带父子关系的 span（任务为根、Agent 节点为子、llm_call/降级/返工为 span event），可指向任意 OTLP 后端（Laminar、Langfuse 等）；**本地 JSONL 仍是主路径**，未配置 endpoint 时完全不介入。
- **调用主体可追溯**：API Key 鉴权解析出的 actor 写入 trace 事件与任务记录 —— v2 记了模型/token/成本，唯独缺"是谁在调"。
- **增量关键词索引**：BM25 缓存分词结果，追加文档只对新增内容分词（引擎因 IDF 依赖全语料仍需重建，这是 `rank_bm25` 的限制）。
- **HTTP + MCP 双入口**：FastAPI/SSE 面向 Web 前端，官方 MCP SDK v2 暴露 `deep_research` 与只读 `kb_search`，供 LLM 客户端发现和调用。
- **容器化交付与 CI**：多阶段镜像、非 root 运行、健康检查、持久化卷与自动建库组成一键启动链；GitHub Actions 使用锁文件执行 Ruff 和离线 pytest。
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
                              ├─ Embedding → Milvus（可切 Chroma）
                              └─ jieba → BM25 Index

在线查询：Researcher → tools/kb_search.py → Milvus + BM25 → RRF → 可选 Rerank
               └─ 本地检索失败或分数不足 → tools/web_search.py

任务收尾：backend/streaming.py → db/persistence.py → PostgreSQL
          （research_tasks / sub_questions / reports / citations）

Planner / Researcher / Critic / Writer
 └─ core/llm.py → 统一模型调用、重试、token 与估算成本记录

LLM 客户端 → MCP stdio → mcp_server/server.py
                         ├─ deep_research → 同一 LangGraph 工作流
                         └─ kb_search → tools/kb_search.py → 同一 RAG 流水线
```

图中的三类持久化职责不同，互相不能替代：

| 存的东西 | 存在哪 | 用来干什么 |
|---|---|---|
| LangGraph 状态 | SQLite Checkpoint | 继续未完成的任务 |
| 运行事件、token、耗时 | JSONL trace | 排查问题、汇总用量 |
| 业务数据（任务/报告/引用） | PostgreSQL | 事后查询、统计、追溯引用名次 |

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

# 可选：启用接口鉴权时，两处 key 要一致；冒号后是写入 trace/任务记录的 actor
API_KEYS=local-demo-key:本地前端
FRONTEND_API_KEY=local-demo-key

EMBEDDING_BACKEND=fastembed
RERANK_BACKEND=onnx
TRACE_ENABLED=true
```

完整配置及默认值见 [`.env.example`](.env.example)。首次使用 FastEmbed 或 ONNX rerank 时可能需要下载本地模型。

### 4. 构建本地知识库

```bash
uv run python -m rag.index_cli --dir data/kb
```

向量索引默认写入 Milvus（`VECTOR_BACKEND=chroma` 时写 `data/chroma/`），关键词索引写 `data/bm25/`。它们都是可重建的运行时数据，不进入 Git。建库结束后应确认输出中的向量数量与 BM25 数量都大于 0。

本机没有 Docker 时，把 `MILVUS_ENDPOINT` 指向一个本地文件（如 `data/milvus_lite.db`）即可用 Milvus Lite 跑通同一套代码，无需起服务。

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

`API_KEYS` 留空时保持默认免鉴权行为。启用后，Streamlit 会从 `FRONTEND_API_KEY` 读取单个客户端凭据并发送 `X-API-Key`；它不会读取后端完整的 key 与 actor 映射。若两处不匹配，页面会明确提示配置问题。

### 6. 使用 Docker Compose 一键启动

先准备 `.env`，再启动整个系统：

```bash
docker compose up --build
```

Compose 会按以下顺序执行：

```text
etcd + minio  ──healthy──→ milvus ──healthy──┐
                                             ├─→ indexer 全量构建 Milvus + BM25 ──成功──┐
postgres ──healthy──→ migrate（alembic upgrade head）──成功──────────────────────────┤
                                                                                     └─→ backend
                                                                                          └─ /health 通过 → frontend
```

`migrate` 是一次性任务，跑完即退出。**不把 alembic 放进 backend 的启动脚本**，是为了多副本时不会有几个进程同时改表结构。

浏览器仍访问 `http://127.0.0.1:8501`。Milvus 数据、BM25、PostgreSQL 数据、SQLite checkpoint、JSONL trace 与模型缓存位于 Docker 命名卷，重建容器不会丢失；`docker compose down` 只停止并移除容器与网络，不删除这些卷。

当前首版会在每次 `up` 时全量重建 44 篇小型产品知识库。这样牺牲少量启动时间，换取了索引与当前语料、切分配置始终一致，也避免额外维护一套“索引是否过期”判断逻辑。

如果 Windows 上的项目绝对路径包含中文，个别 Docker Desktop / Compose Bake 版本可能在 `docker compose build` 阶段报告 `non-printable ASCII` header。可以把仓库放到纯英文路径，或用以下等价方式绕过 Compose 的 Bake 层：

```bash
docker buildx build --load -f Dockerfile.backend -t deepresearch-agent-backend:local .
docker buildx build --load -f Dockerfile.frontend -t deepresearch-agent-frontend:local .
docker compose up --no-build
```

### 7. 本地执行 CI 同款检查

```bash
uv sync --frozen --group dev
uv run --frozen ruff check .
uv run --frozen python -m pytest -m "not live"
```

GitHub Actions 在 push 与 pull request 时执行同样的质量门禁。工作流不会注入 DeepSeek、Tavily 或其他私密 Key；需要真实网络或付费服务的测试必须显式标记 `@pytest.mark.live`，并被 CI 排除。

## API、SSE 与断点恢复

新任务最安全的做法是只传主题，让后端生成唯一 `thread_id`。以下命令适用于 Bash / Git Bash：

如果后端配置了 `API_KEYS`，手工调用还需增加 `-H "X-API-Key: <key>"`；未启用鉴权时可省略。

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

配置 `DATABASE_URL` 后还可查询任务历史：

| 接口 | 作用 |
|---|---|
| `GET /tasks` | 按开始时间倒序列出任务 |
| `GET /tasks/{thread_id}` | 查询任务状态与子问题 |
| `GET /tasks/{thread_id}/report` | 读取最新报告与带名次的引用 |

后端在返回 `start` SSE 前按 `thread_id` 写入 `running`；正常结束更新为 `completed`，图执行或 checkpoint 恢复异常则更新为 `failed`。恢复同一任务会更新原记录，不新增重复任务。未配置数据库时，这三个历史接口返回 503，研究主流程仍可运行。

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

## MCP Server

项目使用官方 MCP Python SDK v2，以 stdio 暴露两个工具：

| 工具 | 适用场景 | 主要输入 | 结构化输出 |
|---|---|---|---|
| `deep_research` | 需要完整研究和 Markdown 报告 | `topic`；可选 `thread_id`、`resume` | 报告、状态、质量/返工摘要、trace usage、可恢复任务 ID |
| `kb_search` | 只查询本地知识库 | `query`、`top_n`（1—20） | 命中片段、来源、排序分、fallback confidence、trace ID |

`deep_research` 可能调用 DeepSeek 和联网搜索并产生费用；`kb_search` 不调用 LLM 或 Web。两者都复用现有实现：MCP 层不会重新编排 Agent，也不会直接 import `rag/` 内部模块。

仓库已提供项目级 [`.mcp.json`](.mcp.json)。在项目目录启动 Claude Code，批准该配置后可通过 `/mcp` 查看 `deep_research` 和 `kb_search`：

```bash
claude
```

如需手动添加同一 stdio Server，可执行：

```bash
claude mcp add --transport stdio --scope project deepresearch-agent -- \
  uv --directory . run python -m mcp_server.server
```

其他支持 stdio MCP 的客户端可以使用等价配置：

```json
{
  "mcpServers": {
    "deepresearch-agent": {
      "type": "stdio",
      "command": "uv",
      "args": ["--directory", ".", "run", "python", "-m", "mcp_server.server"]
    }
  }
}
```

Server 也可以直接启动用于协议调试；该命令会等待 MCP 客户端通过 stdin/stdout 通信，并不是普通交互式 CLI：

```bash
uv run python -m mcp_server.server
```

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
210 passed
```

这组测试覆盖配置、统一 LLM 入口、trace、检索流水线、公开数据转换、R1—R4 runner、报告重算、四个 Agent、反思回环、并发边界、SSE、MCP schema/协议入口、Docker/Compose/CI 配置契约、前端状态和关闭后重开的 SQLite 恢复。它证明实现满足这些确定性场景，**不等于**真实模型准确率、联网稳定性或生产性能；真实检索数字来自单独保存的结构化评测 raw。

| 维度 | 当前已有证据 | 现在可以得出的结论 | 边界 / 可选后续 |
|---|---|---|---|
| 功能回归（v2 基线） | 148 项离线测试 | v2 约定功能路径可重复通过 | 真实 LLM / 搜索端到端完成率 |
| 检索链路 | 100 题公开 qrels、1,664 passage、400 条结构化观测，六项指标 | 混合检索与重排在该基准上均未取得普遍收益 | 产品知识库外推与 reranker 模型匹配分析 |
| 并发研究 | fake IO + P1/P2 固定任务 runner 测试 | 并发上限、配对任务和续跑边界生效 | 相同真实任务的串行 / 并行耗时对照 |
| Critic 回环 | 确定性图场景 + Q1/Q2 runner 测试 | 返工路由与 Critic 独立开关生效 | 15 题两轮的完成率与质量变化 |
| 成本观测 | LLM / trace 汇总测试 | token 与配置价格估算链路可追踪 | 固定评测集上的平均 token、成本和耗时 |
| MCP 接口 | 官方 SDK 客户端进程内协议测试 + 真实 stdio 子进程握手/调用 | 两个工具可发现，schema 可读，`kb_search` 返回结构化结果 | Claude Code 发送本地工具结果前仍需作者显式授权 |
| 功能回归（v3） | 210 项离线测试（148 基线 + 62 新增） | Milvus 适配器、数据库层、OTel 导出、增量索引、鉴权、前端凭据透传与任务生命周期在约定场景下可重复通过 | — |
| 向量库迁移 | 100 题公开基准上 R1/R2/R3 共 18 项指标，Milvus 与 Chroma 记录逐项一致（最大偏差 0.000045，即报告的四位小数精度） | 换向量库没有改变检索结果，迁移正确 | R4 未重跑：它只重排 R3 的同一候选集，且需下载 cross-encoder 权重 |
| 数据库层 | 11 项 db 测试（SQLite）；真实 PostgreSQL 容器完成两版 Alembic upgrade，`GET /tasks` 返回 200 | CRUD 语义在 SQLite 可重复通过；PostgreSQL 建表、连接和查询入口已在本机跑通 | 同一套 db 测试在真实 PostgreSQL 上的自动回归仍由 CI `database` job 承担，远端未跑过不声明绿色 |
| OTel 导出 | 12 项测试，用官方 `InMemorySpanExporter` 走真实 SDK 读回 span | span 父子关系、无配对事件挂载、异常兜底、多 trace 隔离均正确 | **未验证真实后端能否收下这些 span** —— 需要一个真实 OTLP endpoint；看板截图同理尚未产出 |
| 增量索引 | 8 项测试，含"增量结果与全量重建等价"与分词调用次数断言 | 追加不改变排序与分数，且旧文档不重复分词 | 只在 128 切片规模验证；这是增量分词不是增量 BM25 |
| 接口、鉴权与生命周期 | 17 项测试（401/404/503、前端请求头、401 页面提示、actor 落库、`running/completed/failed`、恢复幂等） | 调用主体、前后端鉴权契约、任务状态与历史查询接口行为符合约定 | 明文 key、无轮换、无权限分级，不是生产级方案 |
| 容器交付 | **v3 本机实跑**：PostgreSQL 两版迁移成功；44 篇 → 128 chunk，Milvus / BM25 各 128 条；backend / frontend 均 healthy 且 HTTP 200 | `postgres → migrate` 与 `etcd + minio → milvus → indexer → backend → frontend` 两条门控链成立 | 当前 `.env` 未启用 API 鉴权，未运行真实付费研究；中文路径需用文档中的 buildx 绕过 Compose Bake |

Phase 13 已接入公开中文 `C-MTEB/T2Reranking`：固定抽取 100 个 query，将 positive 与 hard negative 合并为 1,664 个 passage 的共享池，并直接沿用公开 qrels。正式 R 轨生成 400 条结构化观测：

| 组 | Candidate Recall@20 | Hit@5 | MRR@5 | Recall@5 | nDCG@5 | MAP@20 |
|---|---:|---:|---:|---:|---:|---:|
| R1 向量 | **93.64%** | **96.00%** | 0.7238 | **45.46%** | 0.6254 | **0.6204** |
| R2 BM25 | 85.65% | 94.00% | 0.7575 | 40.81% | 0.5961 | 0.5538 |
| R3 RRF | 92.27% | 95.00% | **0.7777** | 43.89% | **0.6359** | 0.6144 |
| R4 + rerank | 92.27% | 93.00% | 0.7377 | 44.27% | 0.6255 | 0.6140 |

该数据集平均每个 query 有 7.55 个正例，首命中型的 Hit@5 / MRR@5 会较早饱和，因此补齐了 Recall@5、nDCG@5 与官方口径 MAP@20 再判读（补测时前三列数值完全复现）。当前结论：

- **混合检索是权衡而非普遍收益** —— R3 相较 R1 顶部加权指标更好（MRR@5 +0.0538、nDCG@5 +0.0105），但覆盖类指标全面变差（Recall@5 -1.56 个百分点、MAP@20 -0.0059）。机制是 Top-20 名额有限，较弱的 BM25 通道在融合时挤占了向量候选。
- **当前 reranker 未观察到收益** —— R4 相较 R3 的 MAP@20 为 -0.0004，基本持平；首命中指标显示的 -2.00 个百分点夸大了退化幅度。
- **单向量基线在本基准上最强**，四项指标最优。结论不好看，但如实保留。

因此双通道与重排都保留为可切换开关，不默认启用。完整证据与边界见 [`eval/reports/comparison.md`](eval/reports/comparison.md) 和 [`EVAL.md`](EVAL.md)。

## 技术决策摘要

| 决策 | 为什么这样做 | 代价 / 边界 |
|---|---|---|
| LangGraph 条件图 | 显式表达节点、循环、条件边与恢复点 | 引入框架概念和状态 schema 维护成本 |
| Milvus + BM25 + RRF | 同时覆盖语义相近与关键词精确匹配，RRF 无需直接比较异构分数 | 需要维护两套索引并用评测校准参数 |
| Milvus 取代 Chroma，但保留 Chroma 分支 | 企业侧向量库更常见；保留旧后端是为了做 A/B 对照，验证换库没有改变检索结果 | **本项目语料只有百余 chunk，Chroma 完全够用**——换库是技术栈对口而非性能需要，代价是多三个容器（milvus + etcd + minio） |
| PostgreSQL + SQLAlchemy + Alembic | 任务、报告与引用需要事后查询和追溯；有迁移脚本才能安全改表结构 | 多一个外部依赖；`DATABASE_URL` 留空时整层跳过，保住无依赖离线回归 |
| `citations` 存检索名次 | 事后要能回答「当时为什么是这条排第一」——索引重建后就再也复原不了 | 上游 `Citation` 暂未携带分数，`retrieval_score` 取不到时留空而非填 0 |
| 落库失败只告警不中断 | 研究助手丢一条记录不是灾难，但因落库失败让用户拿不到报告不可接受 | 需要靠日志发现写入异常；合规类场景应改为写不进就中止 |
| 本地优先、低分联网 | 优先利用可控语料，仅在证据不足时承担联网成本 | 阈值目前仍需正式评测校准 |
| `asyncio` + `Semaphore` | 并发独立 IO 子问题，同时限制资源和 API 压力 | 必须配套超时、限流与异常隔离 |
| Critic 定向返工 | 将“研究证据不足”转换为可检索的明确缺口 | Critic 分数不是第三方质量结论，需对照实验验证收益 |
| SQLite Checkpoint + JSONL trace | 分别处理状态恢复与事件审计 | 产生两类运行时数据，当前更适合单机演示 |
| 统一 LLM Gateway | 集中重试、计量、价格估算和 trace | 公共入口成为需要重点测试的关键模块 |
| 离线 fake embedding | 自动化回归不依赖网络、模型下载和 API Key | 只能验证确定性逻辑，不能代表真实检索质量 |
| 官方 MCP SDK v2 | 用标准工具发现、JSON Schema 与 stdio transport 服务 LLM 客户端 | `deep_research` 可能产生模型/联网费用；项目包使用 `mcp_server` 避免遮蔽第三方 `mcp` |
| 多阶段镜像 + Compose 启动门控 | 依赖锁定在构建阶段，运行阶段非 root；索引完成和后端健康后才启动下游 | 前后端当前复用完整运行依赖，镜像约 324 MB；小语料每次启动全量重建索引 |
| Ruff + 离线 pytest CI | push / PR 自动阻止未使用导入、导入漂移和功能回归，且不需要任何 API Key | 真实联网与付费路径必须在本地以 `live` 测试或 smoke 单独验证 |

## 目录结构

```text
agents/       ResearchState、LangGraph 图与四个 Agent 节点
backend/      FastAPI、异步 SSE 与 checkpoint 恢复入口
core/         集中配置、LLM Gateway、成本估算、trace、checkpoint
data/kb/      Git 可追踪的本地知识库源语料
eval/         公开检索集转换、端到端题集、指标与报告
frontend/     Streamlit 任务进度与报告页面
.github/      push / PR 的 Ruff + 离线 pytest 工作流
mcp_server/   MCPServer、deep_research / kb_search 与 stdio 入口
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
- [`RESUME_MAPPING.md`](RESUME_MAPPING.md)：真实证据、简历表达与高频追问的逐条映射。
- [`INTERVIEW_GUIDE.md`](INTERVIEW_GUIDE.md)：按模块组织的中文口述稿与面试前自检。
- [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md)：作者自行执行的密钥检查、仓库重命名、推送与 GitHub 展示清单。

## 已知边界与后续路线

- 当前标准测试不调用真实 DeepSeek、Tavily 或远程 embedding，联网效果仍需 live 验证。
- 引用由工作流和 Prompt 组织，尚未实现“证据是否语义支持结论”的自动事实核验。
- SQLite checkpoint 面向单机演示；多实例部署需要外部共享存储。
- Streamlit 尚未提供 checkpoint 恢复 UI，当前需通过 API 恢复。
- P/Q 真实付费对照已转为秋招后可选实验，未运行就不声明并发加速或 Critic 质量收益。
- Phase 14 的官方 MCP 客户端与 stdio 链路已验证；Claude Code 实际工具调用会把工具结果发送给外部模型，保留为作者显式授权后的可选验收。
- v3 Docker 镜像和 Compose 启动链已在本机验证，包含 Milvus、PostgreSQL 迁移、双路建库及前后端健康；真实付费研究和鉴权开启态未纳入本轮容器验收。GitHub Actions 的绿色徽章仍需作者提交并推送后由远端运行生成。
