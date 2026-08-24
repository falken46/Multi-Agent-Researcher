# DeepResearch Agent - 架构设计文档 (v2)

> 本文档描述 v2 目标架构。v1 架构见 git 历史 commit `4067aa4`。

---

## 1. 分层架构

```
+-------------------------------------------------------------+
|                    Streamlit 前端                            |
|      主题输入 / 进度展示 / 报告下载 / Trace 面板              |
+---------------------------+---------------------------------+
                            | HTTP + SSE
+---------------------------v---------------------------------+
|                     接口层 (Interface)                       |
|   backend/api.py  FastAPI          mcp/server.py  MCP Server |
+---------------------------+---------------------------------+
                            |
+---------------------------v---------------------------------+
|                   编排层 (Orchestration)                     |
|                   agents/graph.py  LangGraph                 |
|                                                              |
|   +---------+   +------------+   +--------+   +--------+     |
|   | Planner |-->| Researcher |-->| Critic |-->| Writer |     |
|   +---------+   +------------+   +---+----+   +--------+     |
|                      ^               |                       |
|                      |   条件边：质量不达标回退（最多 2 轮）  |
|                      +---------------+                       |
+---------------------------+---------------------------------+
                            |
+---------------------------v---------------------------------+
|                     工具层 (Tools)                           |
|   kb_search          web_search          web_fetch           |
+---------------------------+---------------------------------+
                            |
+---------------------------v---------------------------------+
|                   检索层 (Retrieval / RAG)                   |
|                                                              |
|   loader -> splitter -> embeddings -> vectorstore (Chroma)   |
|                      -> bm25 index                           |
|                                                              |
|   查询：向量召回 + BM25 召回 -> RRF 融合 -> Rerank -> Top-N   |
+---------------------------+---------------------------------+
                            |
+---------------------------v---------------------------------+
|                 基础设施层 (Core Infra)                      |
|   config.py 配置   llm.py 统一客户端   trace.py 调用链       |
|                    costs.py 成本换算                         |
+-------------------------------------------------------------+
```

**分层原则**：上层可以调用下层，下层不得反向依赖。工具层是编排层与检索层之间的唯一通道，Agent 不直接 import `rag/` 内部模块。

---

## 2. 核心模块

### 2.1 基础设施层 `core/`

#### `core/config.py`

集中管理所有配置项，使用 `pydantic-settings` 从 `.env` 读取并做类型校验。禁止在业务代码中散落 `os.getenv`。

关键配置项：

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NAME` | deepseek-v4-flash | 主模型 |
| `EMBEDDING_BACKEND` | fastembed | `fastembed` / `remote` / `fake` |
| `EMBEDDING_MODEL` | BAAI/bge-small-zh-v1.5 | 中文小模型 |
| `RETRIEVAL_TOP_K` | 20 | 单通道召回数量 |
| `RERANK_TOP_N` | 5 | 重排后进入上下文的数量 |
| `RRF_K` | 60 | RRF 平滑常数 |
| `KB_SCORE_THRESHOLD` | 0.35 | 低于此分数触发联网降级 |
| `MAX_REVISION` | 2 | 反思回环硬上限 |
| `QUALITY_THRESHOLD` | 0.7 | Critic 通过分数线 |
| `RESEARCH_CONCURRENCY` | 3 | 子问题并发上限 |
| `LLM_TIMEOUT` | 60 | 单次 LLM 调用超时（秒） |

#### `core/llm.py`

**职责**：所有 LLM 调用的唯一入口。

统一封装：

1. 客户端构造（DeepSeek OpenAI 兼容格式）
2. 超时与重试（指数退避，最多 3 次；仅对超时与 5xx 重试，4xx 不重试）
3. **token 计量**：从 `response.usage` 提取 prompt / completion tokens
4. **自动写 trace**：每次调用记录 trace_id、node、model、tokens、latency、cost

对外接口：

```python
def chat(messages, *, node: str, trace_id: str, json_mode: bool = False) -> LLMResult
async def achat(messages, *, node: str, trace_id: str, json_mode: bool = False) -> LLMResult
```

`LLMResult` 包含 `content`、`usage`、`latency_ms`、`cost`。

> **设计理由**：v1 中 Planner 与 Researcher 各自构造 OpenAI client、各自读环境变量、各自处理异常，重复且无法统一计量。统一入口是实现"成本可观测"的前提。

#### `core/trace.py`

**职责**：记录一次任务的完整调用链。

- 每个任务分配 `trace_id`（uuid4）
- 事件以 JSONL 追加写入 `traces/{date}/{trace_id}.jsonl`，避免并发写冲突
- 事件类型：`node_start` / `node_end` / `llm_call` / `tool_call` / `retrieval` / `fallback` / `revision`
- 提供 `summarize(trace_id)` 聚合出：总 token、总成本、总耗时、各节点耗时占比、降级次数、反思轮次

> **设计理由**：选 JSONL 而非直接写数据库，是因为追加写天然并发安全、便于 grep 排查、且可被评测脚本按行直接解析。聚合查询量小，读时计算即可。

#### `core/costs.py`

维护模型价格表（单位：元 / 百万 token，输入与输出分别计价），提供 `estimate(model, prompt_tokens, completion_tokens) -> float`。

> 价格表从配置文件读取而非硬编码，因为模型定价会变；实际数值需在实现时从服务商官方定价页核对填入。

---

### 2.2 检索层 `rag/`

#### 建库流水线

```
文档目录
  |
  v
loader.py      按扩展名分派解析（.md / .txt / .pdf），产出 Document{text, metadata}
  |
  v
splitter.py    递归字符切分，中文优先按句号、问号、感叹号、换行断句
  |            chunk_size=500, chunk_overlap=80（可配置）
  |            每个切片携带 doc_id / chunk_index / source_path
  |
  |---> embeddings.py -> vectorstore.py   写入 Chroma 持久化目录
  |
  |---> bm25.py                           jieba 分词后建 BM25 索引，pickle 落盘
```

#### 查询流水线

```
query
  |
  |--> 向量通道：embed(query) -> Chroma 相似度检索 -> Top-K 候选
  |
  |--> 关键词通道：jieba 分词 -> BM25 打分 -> Top-K 候选
  |
  v
hybrid.py   RRF 融合
            score(d) = sum over channels of  1 / (RRF_K + rank_channel(d))
  |
  v
rerank.py   对融合后 Top-M 候选重排，取 Top-N
  |
  v
返回 RetrievalResult[]：{text, source, chunk_index, score, channel}
```

#### 模块职责

| 模块 | 职责 | 关键设计 |
|------|------|----------|
| `loader.py` | 文档解析 | 解析失败不中断建库，记录到失败清单 |
| `splitter.py` | 切分 | 中文标点优先，避免英文分句规则切碎中文 |
| `embeddings.py` | 向量化 | **可插拔后端**：`fastembed`（ONNX 本地）/ `remote`（HTTP API）/ `fake`（测试用确定性哈希向量） |
| `vectorstore.py` | 向量库封装 | 只暴露 `add` / `query` / `count`，屏蔽 Chroma 细节，便于替换 |
| `bm25.py` | 关键词检索 | jieba 精确模式分词；索引与向量库共享同一份 chunk id |
| `hybrid.py` | 融合 | RRF，不做分数归一化 |
| `rerank.py` | 重排 | 两种实现可切换：ONNX cross-encoder / LLM rerank |
| `pipeline.py` | 对外入口 | `build_index(dir)` 与 `search(query, top_n)` |
| `index_cli.py` | 建库命令行 | `python -m rag.index_cli --dir data/kb` |

> **为什么用 RRF 而不是加权分数融合**：向量相似度与 BM25 分数量纲不同，直接加权需要归一化，而归一化对分数分布敏感、跨查询不稳定。RRF 只用排名不用绝对分数，无需调参即可稳定工作，是混合检索的常见默认解。

> **为什么 embedding 后端要可插拔**：一是测试需要确定性、零网络依赖的假实现；二是本地 ONNX 模型与远程 API 各有适用场景（离线 vs 无本地算力），抽象一层可在不改业务代码的前提下切换；三是评测时需要对比不同 embedding 的召回差异。

---

### 2.3 工具层 `tools/`

#### `tools/kb_search.py`（新增）

```python
def kb_search(query: str, top_n: int = 5) -> KBSearchResult
```

返回结构包含 `hits`（检索结果列表）与 `max_score`。Researcher 根据 `max_score` 与 `KB_SCORE_THRESHOLD` 决定是否降级到联网检索。

> **为什么降级判断放在 Researcher 而不是 kb_search 内部**：工具层只做检索、不做策略决策，保持"工具只做 IO、不含 Agent 逻辑"的既有铁律。降级属于编排策略，是 Agent 的职责。

---

### 2.4 编排层 `agents/`

#### 状态定义 `agents/state.py`（v2 扩展）

```python
class Citation(TypedDict):
    source: str        # URL 或本地文件路径
    origin: str        # "kb" | "web"
    snippet: str

class ResearchState(TypedDict):
    # v1 已有
    topic: str
    sub_questions: list[str]
    research_results: dict[str, str]
    final_report: str
    errors: list[str]
    retry_count: int
    # v2 新增
    citations: dict[str, list[Citation]]   # 子问题 -> 来源列表
    critique: str                          # Critic 文字意见
    quality_score: float                   # Critic 评分 0-1
    missing_aspects: list[str]             # 需补查方向
    revision_count: int                    # 反思轮次
    trace_id: str                          # 全链路追踪 ID
    usage: dict                            # 累计 token 与成本
```

> **`retry_count` 与 `revision_count` 为什么要分开**：前者是"技术失败重试"（检索抛异常、模型返回空），后者是"质量不达标的主动返工"。两者混用会导致一次网络抖动吃掉返工配额，因此必须分开计数，各自独立设上限。

#### 图拓扑 `agents/graph.py`（v2）

```
START -> planner -> researcher -> [should_continue] --+--> critic
                        ^                             |
                        |                             +--> writer  （检索全失败，产出降级报告）
                        |
                        |            [should_revise]
                        +--------------- critic ------+--> writer -> END
                            （质量不足且未超上限时回退）
```

两个条件函数：

| 函数 | 位置 | 判断逻辑 |
|------|------|----------|
| `should_continue` | researcher 之后 | 有结果 → critic；无结果且未超 `retry_count` 上限 → 重试 researcher；无结果且已超上限 → writer（降级报告） |
| `should_revise` | critic 之后 | `quality_score >= QUALITY_THRESHOLD` → writer；低于阈值且 `revision_count < MAX_REVISION` → 回退 researcher；否则 → writer |

**防死循环三重保险**：

1. `revision_count` 硬上限（`MAX_REVISION`，默认 2）
2. 回退时把 `missing_aspects` 写入状态，Researcher 只针对缺失方向补查，不重复全量检索
3. 若两轮返工后分数**没有提升**，直接进入 Writer，不再尝试

> **为什么 Critic 只评审不改写**：同一个节点既评价又修改，会倾向于给自己的修改打高分，评分随之失去意义；分离之后 `quality_score` 才能作为评测指标使用。

#### `agents/critic.py`（新增）

- 使用 `response_format=json_object` 强制结构化输出
- 输出 schema：`{"quality_score": float, "critique": str, "missing_aspects": [str]}`
- 解析失败时降级：给出中性分数并记录错误，**不阻断主流程**

#### `agents/researcher.py`（改造要点）

```python
async def researcher_node(state) -> dict:
    semaphore = asyncio.Semaphore(config.RESEARCH_CONCURRENCY)
    targets = state["missing_aspects"] or state["sub_questions"]   # 返工时只查缺口
    results = await asyncio.gather(
        *(_research_one(q, semaphore, state["trace_id"]) for q in targets),
        return_exceptions=True,
    )
```

- `return_exceptions=True` 保证单个子问题失败不炸整体
- `Semaphore` 控制并发，避免触发搜索 API 限流
- 每个子问题内部：`kb_search` → 判断 `max_score` → 必要时 `web_search` → LLM 摘要 → 产出 `citations`

---

### 2.5 接口层

#### `backend/api.py`（沿用 v1，扩展事件）

SSE 事件类型新增：`critic_start` / `critic_done` / `revision` / `fallback` / `usage`，前端据此展示反思过程与成本。

#### `mcp/server.py`（新增）

基于 FastMCP 暴露两个工具：

| 工具 | 参数 | 返回 |
|------|------|------|
| `deep_research` | `topic: str` | 完整 Markdown 报告 |
| `kb_search` | `query: str, top_n: int` | 检索结果列表 |

> **为什么要做 MCP Server**：实习经历中作者是 MCP 工具的**调用方**，做 Server 才补上**生产方**视角。面试中"MCP 与普通 HTTP API 的区别"是高频问题，亲手实现过才答得清楚 —— 差异在于 MCP 面向 LLM 客户端标准化了工具描述与发现方式，并由客户端统一管理连接生命周期与权限，而普通 HTTP API 的接口契约由业务方各自定义、需要为每个客户端单独适配。

---

## 3. 数据流：一次完整任务

```
1.  前端 POST /research {topic}
2.  后端生成 trace_id，创建初始 state，异步启动 LangGraph
3.  planner_node    -> core.llm.achat(json_mode=True) -> sub_questions[]
                       写 trace: node_start / llm_call / node_end
4.  researcher_node -> asyncio.gather 并发处理子问题
                       单个子问题: kb_search -> [是否降级] -> web_search -> achat 摘要
                       写 trace: retrieval / fallback / llm_call
5.  should_continue -> critic
6.  critic_node     -> achat(json_mode=True) -> quality_score / missing_aspects
7.  should_revise   -> 分数不足且未超上限 -> 回到步骤 4（只查 missing_aspects）
                    -> 否则 -> 步骤 8
8.  writer_node     -> achat -> final_report（带角标引用）
9.  trace.summarize(trace_id) -> usage 汇总，通过 SSE 推送前端
10. 前端渲染报告 + Trace 面板
```

---

## 4. 目录结构（v2）

```
deepresearch-agent/
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── pyproject.toml / requirements.txt
├── .env.example
├── README.md
├── PRD.md / ARCHITECTURE.md / TECH_STACK.md / TASKS.md
├── TESTING.md / EVAL.md / OBSERVABILITY.md / RESUME_MAPPING.md
│
├── core/                      # 【v2 新增】基础设施层
│   ├── config.py
│   ├── llm.py
│   ├── trace.py
│   └── costs.py
│
├── rag/                       # 【v2 新增】检索层
│   ├── loader.py
│   ├── splitter.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── bm25.py
│   ├── hybrid.py
│   ├── rerank.py
│   ├── pipeline.py
│   └── index_cli.py
│
├── agents/
│   ├── state.py               # v2 扩展
│   ├── graph.py               # v2 改造
│   ├── planner.py
│   ├── researcher.py          # v2 改造：异步 + 双通道
│   ├── critic.py              # 【v2 新增】
│   ├── writer.py
│   └── prompt_loader.py
│
├── tools/
│   ├── web_search.py
│   ├── web_fetch.py
│   └── kb_search.py           # 【v2 新增】
│
├── prompts/
│   ├── planner_system.md
│   ├── researcher_system.md
│   ├── critic_system.md       # 【v2 新增】
│   └── writer_system.md
│
├── backend/
│   ├── api.py
│   └── streaming.py
│
├── frontend/
│   └── app.py
│
├── mcp/                       # 【v2 新增】
│   └── server.py
│
├── eval/                      # 【v2 新增】评测层
│   ├── dataset/qa.jsonl
│   ├── runner.py
│   ├── metrics.py
│   ├── report.py
│   └── reports/               # 生成的对照表
│
├── data/
│   ├── kb/                    # 知识库原始文档
│   └── chroma/                # 向量库持久化
│
├── traces/                    # trace JSONL 落盘
│
└── tests/
```

---

## 5. 关键技术决策（ADR）

> 面试高频追问区。每条都要能说出"为什么不选另一个"。

| # | 决策点 | 选择 | 理由 | 放弃的方案及原因 |
|---|--------|------|------|------------------|
| 1 | Agent 框架 | LangGraph | 显式状态机，条件边与循环是一等公民，天然支持 checkpoint | 手写调度：无法体现框架能力；AutoGen：对话式抽象不适合确定性流程 |
| 2 | 向量库 | Chroma | 嵌入式、零运维、Python 原生，Demo 规模足够 | Milvus：需独立部署，本项目数据量用不上；FAISS：缺少元数据过滤 |
| 3 | Embedding | fastembed (ONNX) + bge-small-zh | **不依赖 torch**，镜像体积从 GB 级降到百 MB 级，CPU 推理够快 | sentence-transformers：拖入 torch，Docker 镜像过大 |
| 4 | 混合融合 | RRF | 只用排名不用分数，免归一化、免调参、跨查询稳定 | 加权求和：需分数归一化，对分布敏感 |
| 5 | 重排 | 可切换（ONNX cross-encoder / LLM rerank） | 前者快且便宜，后者零额外模型依赖；对照实验可量化二者差异 | 付费 Rerank API：成本不可控且无法离线 |
| 6 | 反思机制 | 独立 Critic 节点 + 条件边 | 评审与改写分离，`quality_score` 才可用作指标 | 在 Writer 里自评：既当运动员又当裁判 |
| 7 | 并发 | asyncio + Semaphore | 子问题相互独立且为 IO 密集，异步收益最大 | 多线程：GIL 下无优势且难与 FastAPI 配合 |
| 8 | Trace 存储 | JSONL 追加写 | 并发安全、可 grep、评测脚本可直接解析 | SQLite：并发写需加锁，收益不匹配复杂度 |
| 9 | 配置管理 | pydantic-settings 集中管理 | 类型校验 + 单一真源，避免 `os.getenv` 散落 | 直接读环境变量：v1 的问题，无校验、易漂移 |
| 10 | Checkpointer | LangGraph SqliteSaver | 官方实现，支持断点续跑与状态回放 | 纯内存：进程退出即丢失 |
| 11 | 对外协议 | HTTP + MCP 双通道 | HTTP 面向前端，MCP 面向 LLM 客户端，覆盖两类消费者 | 仅 HTTP：错失 MCP Server 开发经验 |

---

## 6. Prompt 设计原则（v2 补充）

1. 所有 system prompt 位于 `prompts/*.md`，通过 `load_prompt(name)` 加载（沿用铁律 3）
2. 需要结构化输出的节点（Planner / Critic）一律使用 `response_format=json_object`，并在解析层做**容错降级**而非直接抛错
3. Critic 的 prompt 必须给出明确评分锚点（什么情况给 0.3、什么情况给 0.8），否则分数无法跨任务比较
4. Writer 的 prompt 必须约束引用编号与传入的 citations 严格对应，这是"引用可溯源率"能被程序校验的前提

---

## 7. 演进边界

以下内容明确留给未来，不在 v2 范围内：

- 知识库增量更新与文档去重
- 检索结果缓存层（P2）
- 多用户与任务队列
- 向量库替换为 Milvus 的生产部署形态
