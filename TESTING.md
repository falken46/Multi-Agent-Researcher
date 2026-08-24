# DeepResearch Agent - 测试与验收标准 (v2)

> v1 已有 41 条测试，全部保留并作为**回归红线**。v2 新增模块须达到同等覆盖标准。

---

## 1. 测试分层

| 层级 | 位置 | 是否需要网络 | 在 CI 中运行 |
|------|------|--------------|--------------|
| 单元测试 | `tests/test_*.py` | 否 | ✓ |
| 集成测试 | `tests/test_graph*.py` | 否（mock LLM） | ✓ |
| 端到端测试 | `tests/test_e2e.py` | mock 模式否 | ✓ |
| 真实调用测试 | 标记 `@pytest.mark.live` | 是 | ✗（本地手动跑） |

**核心原则**：CI 中运行的测试**不得依赖任何 API Key 或外网**。需要真实调用的测试一律打 `live` 标记并默认跳过。

```python
# pyproject.toml
[tool.pytest.ini_options]
markers = ["live: 需要真实 API Key 与网络，CI 中跳过"]
addopts = "-m 'not live'"
```

---

## 2. v2 新增模块的测试要求

### 2.1 基础设施层 `core/`

| 模块 | 测试要点 |
|------|----------|
| `config.py` | 缺失必填项时报错清晰；默认值正确；类型转换正确 |
| `costs.py` | 已知 token 数换算结果正确；未知模型不崩溃（返回 0 并告警） |
| `trace.py` | 事件写入后可读回；`summarize()` 聚合正确；并发写不丢事件 |
| `llm.py` | 重试逻辑：超时重试、4xx 不重试；token 从 usage 正确提取；调用后 trace 有记录 |

`llm.py` 的重试测试必须 mock 掉真实 sleep，避免测试变慢：

```python
def test_retry_on_timeout_not_on_4xx(mocker):
    mocker.patch("time.sleep")          # 不真的等待
    ...
```

### 2.2 检索层 `rag/`

> **关键约定**：所有检索层测试使用 `EMBEDDING_BACKEND=fake`。
> `fake` 后端根据文本内容生成确定性哈希向量，保证测试可复现、零网络依赖、毫秒级完成。

| 模块 | 测试要点 |
|------|----------|
| `splitter.py` | 中文长文本按标点切分正确；overlap 生效；元数据（doc_id / chunk_index）正确携带；空文档不崩溃 |
| `embeddings.py` | 三种后端接口一致；`fake` 后端对相同输入返回相同向量 |
| `vectorstore.py` | add 后 count 正确；query 返回条数符合 top_k；元数据可回取 |
| `bm25.py` | 中文分词后能命中关键词；未命中返回空而非报错 |
| `hybrid.py` | **RRF 计算结果与手算一致**（构造两路已知排名，断言融合后顺序） |
| `rerank.py` | 重排后条数等于 top_n；`none` 后端等价于直通 |
| `pipeline.py` | 建库 → 检索链路贯通；单通道失效时可降级 |

`hybrid.py` 是纯函数，最容易写出高价值测试，必须覆盖：

```python
def test_rrf_prefers_doc_ranked_high_in_both_channels():
    vector_ranks = ["d1", "d2", "d3"]
    bm25_ranks   = ["d3", "d1", "d4"]
    fused = rrf_fuse({"vector": vector_ranks, "bm25": bm25_ranks}, k=60)
    assert fused[0] == "d1"      # 两路都靠前
```

### 2.3 编排层 `agents/`

| 模块 | 测试要点 |
|------|----------|
| `critic.py` | JSON 解析成功路径；解析失败时降级为中性分数且不抛出；分数越界被裁剪到 [0,1] |
| `researcher.py` | 并发上限生效；单个子问题异常不影响其他；`missing_aspects` 非空时只查缺口；降级触发条件正确 |
| `graph.py` | `should_revise` 三条分支各一测试；**回退次数不超过 MAX_REVISION**；分数无提升时提前退出 |

其中"回退上限"是防御性测试，必须显式验证：

```python
def test_revision_never_exceeds_hard_limit(mocker):
    # Critic 永远返回 0.1（永不达标）
    mocker.patch("agents.critic.critic_node", return_value={"quality_score": 0.1, ...})
    result = build_graph().invoke(create_initial_state("任意主题"))
    assert result["revision_count"] <= config.MAX_REVISION
```

### 2.4 接口层

| 模块 | 测试要点 |
|------|----------|
| `backend/api.py` | 新增 SSE 事件类型能正确推送；异常时返回结构化错误 |
| `mcp/server.py` | 工具注册成功；参数 schema 正确；调用返回结构符合约定 |

---

## 3. 异步测试约定

v2 大量使用 `asyncio`，统一使用 `pytest-asyncio`：

```python
@pytest.mark.asyncio
async def test_researcher_runs_concurrently(mocker):
    ...
```

并发行为的验证方式：mock 单次检索为固定延时，断言**总耗时明显小于串行耗时之和**，而不是断言具体线程调度顺序（后者不稳定）。

---

## 4. 端到端验收主题

真实调用（`@pytest.mark.live`，本地手动执行），覆盖三类场景：

| # | 主题 | 期望验证点 |
|---|------|-----------|
| 1 | 语料内明确存在的技术问题 | 全程走 kb_search，不触发降级 |
| 2 | 语料完全不涉及的时事问题 | 正确触发降级到 web_search |
| 3 | 一半在语料内的混合问题 | 两个通道都被使用，报告来源含两种 origin |
| 4 | 语料内但表述模糊的问题 | 观察重排是否把正确切片提前 |
| 5 | 刻意宽泛的主题 | 观察 Critic 是否触发返工 |

**验收标准**：5 个主题中至少 4 个产出结构完整、引用可溯源的报告。

---

## 5. 回归红线

每个 Phase 结束前必须执行：

```bash
uv run pytest
```

- v1 原有 41 条测试**必须全部通过**
- 新增模块测试全部通过
- 任何为了让新功能跑通而删改旧测试的行为，都必须在 commit message 中说明原因

---

## 6. CI 配置要点

`.github/workflows/ci.yml`：

1. 使用 `uv sync --group dev` 安装依赖
2. 先跑 `ruff check`，再跑 `pytest`
3. 不注入任何 API Key（验证 CI 测试确实不依赖外部服务）
4. 缓存 uv 依赖以缩短构建时间

> CI 跑绿的前提是所有需要网络的测试都已正确标记为 `live`。若 CI 失败于缺少 Key，说明标记遗漏，属于测试设计问题而非 CI 配置问题。
