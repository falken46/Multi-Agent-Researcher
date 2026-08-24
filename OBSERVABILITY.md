# DeepResearch Agent - 可观测性方案 (OBSERVABILITY)

> 目标：让"这次任务花了多少钱、慢在哪、走了哪条路"三个问题，都能用数据回答。

---

## 1. 为什么单独做这一层

个人 Agent 项目的通病是：出问题只能靠翻终端日志，说不清成本，也说不出瓶颈在哪个节点。

而 Agent 系统的两个真实工程痛点恰恰是：

1. **成本不可控** —— 一次任务调用几十次 LLM，token 消耗完全不透明
2. **瓶颈不明确** —— 端到端两分钟，不知道是检索慢、模型慢，还是串行慢

本层用一套轻量 trace 解决这两个问题，同时为 `EVAL.md` 的效率类指标提供数据源。

> 面试价值：这是**唯一能让你把"降本"和"提速"讲成因果链**的模块。没有它，"耗时下降 40%" 只能靠感觉。

---

## 2. Trace 数据模型

### 2.1 事件 schema

每一行是一个 JSON 事件，追加写入 `traces/{YYYY-MM-DD}/{trace_id}.jsonl`：

```json
{
  "trace_id": "9f1c...",
  "ts": 1756000000.123,
  "event": "llm_call",
  "node": "researcher",
  "payload": {
    "model": "deepseek-v4-flash",
    "prompt_tokens": 1820,
    "completion_tokens": 460,
    "latency_ms": 3120,
    "cost": 0.0042,
    "attempt": 1
  }
}
```

### 2.2 事件类型

| 事件 | 触发时机 | 关键 payload |
|------|----------|--------------|
| `task_start` | 任务开始 | topic |
| `node_start` / `node_end` | 每个 Agent 节点进出 | node 名、耗时 |
| `llm_call` | 每次 LLM 调用（含重试） | model、tokens、latency、cost、attempt |
| `retrieval` | 每次 kb_search | query、各通道候选数、融合后条数、max_score、耗时 |
| `fallback` | 本地召回不足降级联网 | query、max_score、阈值 |
| `tool_call` | web_search / web_fetch | 工具名、参数摘要、耗时、是否成功 |
| `revision` | Critic 触发返工 | 轮次、返工前分数、missing_aspects |
| `error` | 任何捕获的异常 | node、异常类型、消息 |
| `task_end` | 任务结束 | 总耗时、总 token、总成本、最终分数 |

### 2.3 为什么选 JSONL

| 考虑 | 说明 |
|------|------|
| 并发安全 | 追加写单行，多协程写同一文件不需要加锁 |
| 排查友好 | 可直接 `grep fallback traces/**/*.jsonl` 定位所有降级 |
| 评测友好 | `eval/metrics.py` 按行解析即可，无需 ORM |
| 成本低 | 不引入数据库依赖，Docker 镜像不变大 |

**放弃的方案**：SQLite 需处理并发写锁；接入 Langfuse / LangSmith 等托管平台会引入外部依赖与账号，且在离线环境不可用 —— 但在 README 中应说明"若需生产级可观测，此处可平滑替换为 OpenTelemetry 或 Langfuse"，体现了解生态。

---

## 3. 成本模型

`core/config.py` 维护价格表，`core/costs.py` 只负责换算。当前默认值于 2026-08-24 从 [DeepSeek 官方模型与价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing) 核对，单位为人民币 / 百万 token：

| 模型 | 输入（缓存命中） | 输入（缓存未命中） | 输出 |
|------|------------------:|--------------------:|-----:|
| `deepseek-v4-flash` | 0.02 | 1.00 | 2.00 |
| `deepseek-v4-pro` | 0.025 | 3.00 | 6.00 |

价格通过 `MODEL_PRICING` 覆盖，版本日期通过 `MODEL_PRICING_VERSION` 记录；API 未返回缓存明细时，输入 token 按缓存未命中保守估算。

**约定**：

1. 价格表从配置读取，不硬编码在业务逻辑里
2. 未知模型返回 0 并记录 warning，不抛异常（避免因换模型导致任务失败）
3. README 与评测报告中展示成本时，必须同时标注所用价格表版本与日期

> ⚠️ 模型定价会随时间变化。实现时务必从服务商官方定价页核对当前数值再填入，不要凭记忆填写。

---

## 4. 聚合与查询

`core/trace.summarize(trace_id)` 输出：

```python
{
    "total_latency_ms": 68420,
    "total_tokens": 24870,
    "total_cost": 0.0361,
    "llm_calls": 11,
    "by_node": {
        "planner":    {"latency_ms": 2100,  "tokens": 890,   "calls": 1},
        "researcher": {"latency_ms": 51200, "tokens": 18300, "calls": 8},
        "critic":     {"latency_ms": 4300,  "tokens": 2100,  "calls": 1},
        "writer":     {"latency_ms": 10800, "tokens": 3580,  "calls": 1}
    },
    "fallback_count": 2,
    "revision_count": 1,
    "errors": []
}
```

`by_node` 是定位瓶颈的核心：能直接看出 Researcher 占据绝大部分耗时，从而论证"并行化优先于其他优化"这个决策。

---

## 5. 前端 Trace 面板

Streamlit 在报告下方展示：

1. **成本与耗时摘要**：总耗时、总 token、总成本、LLM 调用次数
2. **节点耗时分布**：横向条形图，一眼看出瓶颈节点
3. **执行路径**：是否触发降级、是否触发返工、返工前后分数
4. **错误列表**：本次任务被捕获但未中断流程的错误

> 这个面板同时是 Demo 截图的主要素材 —— 它比报告本身更能体现工程深度，README 中应当放它的截图。

---

## 6. 日志与 trace 的分工

| | 日志（logging） | Trace |
|---|---|---|
| 用途 | 开发期排查、人读 | 结构化度量、机器读 |
| 位置 | 终端 / 文件 | `traces/*.jsonl` |
| 保留 | 短期 | 长期，评测依赖 |
| 内容 | 自由文本 | 固定 schema |

沿用 v1 铁律 5（节点入口打日志），但**指标一律走 trace，不从日志里正则抠数字**。

---

## 7. 这一层支撑的面试问题

| 问题 | 依托的数据 |
|------|-----------|
| 一次任务成本多少？ | `task_end.cost` |
| 系统瓶颈在哪？怎么发现的？ | `summarize().by_node` 的耗时分布 |
| 并行化提升了多少？ | 并行前后的 `total_latency_ms` 对比 |
| 反思机制的代价是什么？ | 触发返工的任务与未触发任务的 token 差异 |
| 降级策略触发得合理吗？ | `fallback` 事件占比与对应 `max_score` 分布 |
| 出错了怎么排查？ | 按 trace_id 拉出完整调用链 |

> 这些问题在面试中出现频率很高，而绝大多数候选人只能定性回答。有 trace 就能定量回答，并且可以现场打开文件给面试官看。
