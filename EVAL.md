# DeepResearch Agent - 评测方案 (EVAL)

> 本文档定义评测集构造、指标定义与对照实验设计。
> **本项目简历上出现的每一个数字，都必须能由本文档描述的流程复现。**

---

## 1. 为什么要做评测

大多数个人 Agent 项目止步于"能跑出结果"，无法回答两个问题：

1. 你的优化到底有没有效果？
2. 效果有多大，是哪一项优化带来的？

评测层的存在，就是把"我做了混合检索"变成"混合检索让召回命中率从 A 提升到 B"。这也是本项目相对同类项目最主要的差异点。

面试中对应的高频追问是"你怎么知道效果变好了"，没有评测就只能回答"感觉上"。

---

## 2. 评测集设计

Phase 13 把检索质量和端到端编排分开，避免 LLM、联网搜索与检索排序同时变化后无法归因。

### 2.1 R 轨：公开检索基准

R 轨使用中文公开数据集 `C-MTEB/T2Reranking`。准备脚本固定抽取 100 个 query，保留每题的 positive 与 hard negative，并把所有抽中题目的 passage 合并成**一个共享语料池**。

```bash
uv run python -m eval.prepare_retrieval_dataset --queries 100 --seed 42
```

生成目录 `eval/dataset/t2_reranking/` 包含：

| 文件 | 用途 |
|------|------|
| `corpus.jsonl` | 所有 query 共用的 passage 候选池 |
| `queries.jsonl` | query、gold passage 与原始候选关系 |
| `qrels.jsonl` | query 到相关 passage 的正式标注 |
| `metadata.json` | 数据来源、抽样参数与实际规模 |

原始数据和生成产物均不进入 Git；仓库只保存转换器、测试和说明。原始 T2Ranking 项目声明 Apache-2.0 许可证，报告需引用数据集原论文与 MTEB/C-MTEB。

T2Ranking 已把文本定义为 passage 检索单元，因此 R 轨不再按产品知识库的 `CHUNK_SIZE` 二次切分。一个 passage 直接对应一个检索 `Chunk` 和一个 qrels ID。

### 2.2 P/Q 轨：端到端编排题集

`eval/dataset/orchestration_qa.jsonl` 共 15 题：本地、联网、混合各 5 题。每题包含固定子问题与 `must_cover` 同义短语组。P 轨取 5 道 KB 题比较串行/并行；Q 轨取全部 15 题比较 Critic 开/关时的完成率、覆盖、引用、耗时、token 和成本。

```json
{
  "id": "O-KB-001",
  "type": "kb",
  "topic": "如何设计不会无限循环的 LangGraph 路由？",
  "sub_questions": ["节点与边分别承担什么职责？"],
  "must_cover": [
    {"id": "routing_boundary", "any_of": ["节点负责状态更新", "边决定执行顺序"]}
  ]
}
```

### 2.3 构造原则

1. **共享候选池**：R 轨的每个 query 都从全部 passage 检索，禁止只对自身正负候选排序。
2. **标注先于实验**：gold 直接来自公开 qrels，不根据本项目的检索结果回填。
3. **检索与生成分轨**：R 轨不调用 LLM 或 Web；P/Q 轨不拿端到端结果冒充纯检索指标。
4. **固定抽样参数**：正式对照统一使用相同 query 数、随机种子与候选上限。
5. **不做额外 hash 验证**：只记录来源、抽样参数和生成规模；数据损坏由 Parquet/JSON 解析错误直接暴露。

---

## 3. 指标定义

### 3.1 检索质量（R 轨）

**候选召回率 Candidate Recall@20**

```
Candidate Recall@20 = Top-20 候选中 gold passage 数 / gold passage 总数
```

**召回命中率 Hit@K**

```
Hit@5 = (Top-5 至少命中一个 gold passage 的样本数) / (R 轨样本总数)
```

**平均倒数排名 MRR**

```
MRR@5 = (1/N) * Σ  1 / rank_of_first_gold_passage
```

**Top-5 召回率 Recall@5**

```
Recall@5 = Top-5 中 gold passage 数 / gold passage 总数
```

**归一化折损累计增益 nDCG@5**（二值增益）

```
DCG@5   = Σ_{rank<=5, 命中}  1 / log2(rank + 1)
IDCG@5  = Σ_{rank=1..min(gold 数, 5)}  1 / log2(rank + 1)
nDCG@5  = DCG@5 / IDCG@5
```

**平均精度均值 MAP@20**

```
AP@20   = (Σ_{rank<=20, 命中} 命中数(rank) / rank) / min(gold 数, 20)
MAP@20  = 全部 query 的 AP@20 均值
```

> **为什么必须加这三项。** 本检索集平均每个 query 有 7.55 个正例（100 query / 755 条 positive 关系）。
> 多正例场景下 Hit@5 只要命中任意一个就记满分，实测已在 93—96% 接近饱和；MRR@5 也只看第一个正例的位置。
> 而 reranker 的实际工作是把**全部**正例顶到负例之上，这件事首命中型指标完全看不见。
> T2Reranking 在 C-MTEB 的官方主指标本就是 MAP，正是出于同一原因。
>
> 分工：Candidate Recall@20 衡量候选生成，Hit@5 / MRR@5 衡量「第一个正例找得到、排得前吗」，
> Recall@5 / nDCG@5 / MAP@20 衡量「多个正例整体排得好吗」。重排无法挽回未进入 Top-20 的 gold，因此需同时报告。
>
> AP 的分母取 `min(gold 数, 20)` 而非 gold 总数：正例数超过截断深度的 query 若用后者将永远无法取到 1.0，
> 组间比较会被各 query 的正例数量分布带偏。nDCG 使用二值增益，因为冻结的 qrels 只标注相关与否，没有分级判断。
>
> 深层指标要求 raw 记录含 `ranked_chunk_ids`（schema_version 2 起落盘完整候选深度排序）。
> 早期 raw 只存最终 Top-5 切片，用它算出的「MAP@20」实际是 AP@5，报告生成器对这种输入 fail-fast。

### 3.2 生成质量（全部样本）

**引用可溯源率 Citation Groundedness**

```
可溯源率 = (报告引用编号能在 citations 中找到对应来源的数量) / (报告引用总数)
```

> 这是**防幻觉指标**。Writer 编造一个不存在的来源编号，此项立刻下降。因为 Writer 的引用编号被约束为与 `citations` 一一对应，该指标可完全程序化校验，无需人工。

**关键点覆盖率 Coverage**

```
覆盖率 = (报告中命中的 must_cover 关键点数) / (must_cover 总数)
```

> 使用关键词匹配 + 同义词表做粗粒度判定。**明确承认这是弱指标**，仅用于组间相对比较，不作为绝对质量结论。

**任务完成率 Completion Rate**

```
完成率 = (成功产出非空报告且无 fatal error 的样本数) / (总样本数)
```

### 3.3 效率与成本（全部样本）

| 指标 | 定义 |
|------|------|
| 平均端到端耗时 | 从任务开始到报告产出的墙钟时间均值 |
| 平均总 token | 单次任务全部 LLM 调用的 prompt + completion token 之和 |
| 平均成本 | 按 `core/costs.py` 价格表换算的单次任务费用 |
| P95 耗时 | 反映长尾，仅看均值会掩盖偶发超时 |

### 3.4 行为指标（v2 特有）

| 指标 | 定义 | 观察目的 |
|------|------|----------|
| 降级触发率 | 触发 `web_search` 降级的子问题占比 | 验证 `KB_SCORE_THRESHOLD` 是否合理 |
| 反思触发率 | `quality_score` 低于阈值而回退的任务占比 | 描述 Critic 的触发行为，过高或过低都需要结合任务检查 |
| Critic 分数变化 | 回退任务在返工后的 Critic 自评分变化均值 | 只作为过程诊断，观察回环是否改变了 Critic 自己的判断 |

> Critic 同时负责给分和触发返工，因此返工后的自评分上升并不是独立的质量证据。Q1/Q2 的主要外部代理指标仍是任务完成率、`must_cover` 覆盖率与引用有效性；Critic 自评分只用于解释回环过程。如果这些代理指标没有改善而 token、成本和耗时上升，应如实报告为当前机制没有可观测净收益。

---

## 4. 对照实验设计

### 4.1 分组

R 轨四组只改变检索组件：

| 组 | 配置 | 独立回答的问题 |
|----|------|------------------|
| **R1** | 仅向量召回 | 语义召回基线怎样 |
| **R2** | 仅 BM25 | 关键词通道怎样 |
| **R3** | 向量 + BM25 + RRF | 混合融合是否优于单通道 |
| **R4** | R3 + ONNX rerank | 精排是否改善前五名顺序 |

端到端编排拆为两个独立对照：

| 轨道 / 组 | 题目 | 并发 | Critic | 唯一变化 |
|---|---|---:|---:|---|
| **P1** | 5 道 KB 题 | 1 | 关 | 串行基线 |
| **P2** | 与 P1 相同 | 3（读取集中配置） | 关 | 只增加并发 |
| **Q1** | 15 道平衡题 | 3 | 关 | 生成质量基线 |
| **Q2** | 与 Q1 相同 | 3 | 开 | 只增加 Critic 回环 |

### 4.2 控制变量

- R1—R4 使用同一个共享 passage 池、query 集、embedding 模型、Top-K 和 Top-N
- P/Q 固定 Planner 子问题；联网证据通过 query-keyed 本地缓存复用
- P 轨第一批跑 1 轮快速取得方向性数字；正式写加速百分比前建议补到 3 轮
- Q 轨每组跑 **2 轮取均值**，R 轨是确定性的，只需运行一轮
- 每组配置与原始结构化结果写入报告产物

### 4.3 指标适用范围

| 指标 | R 轨 | P/Q 轨 |
|------|------|------|
| Candidate Recall@20 / Hit@5 / MRR@5 | ✓ | ✗ |
| 引用可溯源率 / 覆盖率 / 完成率 | ✗ | ✓ |
| 耗时 / token / 成本 | 检索耗时可单列 | ✓ |
| 降级 / 反思行为 | ✗ | ✓ |

> 不适用的指标写 `N/A`，不能写成 0。

---

## 5. 执行流程

```bash
# 1. 下载并生成公开检索子集（一次即可）
uv run python -m eval.prepare_retrieval_dataset --queries 100 --seed 42

# 2. 运行 R1—R4（本地模型，不调用 LLM / Web）
uv run python -m eval.run_retrieval \
  --groups R1 R2 R3 R4 \
  --output eval/reports/raw/retrieval_t2_100_local.jsonl

# 3. 从结构化 raw 与 trace 生成对照报告
uv run python -m eval.report \
  --retrieval-raw eval/reports/raw/retrieval_t2_100_local.jsonl \
  --out eval/reports/comparison.md \
  --overwrite
```

原始结果默认写入 `eval/reports/raw/retrieval_{timestamp}.jsonl`；正式运行也可以用
`--output` 指定稳定文件名，保留原始数据以便复核。

R 轨 runner 为 Chroma 与 BM25 各建一次独立评测索引。每个 query 的双通道结果只计算一次，
R3 与 R4 复用同一批 RRF Top-20 候选；写盘前会校验 Top-5 是候选子集，最后原子生成 JSONL。

### 5.1 已完成的正式 R 轨结果（2026-08-26）

正式运行使用 100 个 query、1,664 个共享 passage、`BAAI/bge-small-zh-v1.5`
本地 embedding 与 `BAAI/bge-reranker-base` 本地 ONNX reranker，共生成 400 条结构化观测：

| 组 | Candidate Recall@20 | Hit@5 | MRR@5 | Recall@5 | nDCG@5 | MAP@20 |
|---|---:|---:|---:|---:|---:|---:|
| R1 向量 | **93.64%** | **96.00%** | 0.7238 | **45.46%** | 0.6254 | **0.6204** |
| R2 BM25 | 85.65% | 94.00% | 0.7575 | 40.81% | 0.5961 | 0.5538 |
| R3 RRF | 92.27% | 95.00% | **0.7777** | 43.89% | **0.6359** | 0.6144 |
| R4 + ONNX rerank | 92.27% | 93.00% | 0.7377 | 44.27% | 0.6255 | 0.6140 |

> 补齐 Recall@5 / nDCG@5 / MAP@20 之前，本表只有前三列。补齐动机与口径见 §3.1。
> 补测时同一份配置重跑，Candidate Recall@20 / Hit@5 / MRR@5 三列**数值完全复现**，
> 确认新增深层指标没有改变既有结论的数据基础。

这组结果支持三个有限结论：

1. **混合检索是权衡，不是普遍收益。** R3 相比 R1 在顶部加权指标上占优（MRR@5 +0.0538、
   nDCG@5 +0.0105），但在覆盖类指标上全面落后：Candidate Recall@20 -1.37、Hit@5 -1.00、
   Recall@5 -1.56 个百分点，**官方口径 MAP@20 也下降 0.0059**。
   机制是 Top-20 名额有限，较弱的 BM25 通道（MAP@20 仅 0.5538）在融合时挤占了向量候选：
   它把个别高相关 passage 顶到最前，同时把另一些相关 passage 挤出候选。
   因此**不能只报 MRR@5 的提升**——那是在多个指标里挑对自己有利的一个。
2. **当前 reranker 接近无效，而非有害。** R4 与 R3 候选完全相同，Candidate Recall@20 不变；
   Recall@5 +0.38 个百分点，nDCG@5 -0.0104，**MAP@20 -0.0004（基本持平）**。
   首命中型指标显示的 Hit@5 -2.00、MRR@5 -0.0400 夸大了退化幅度：
   它们只看第一个正例，而本集平均每 query 有 7.55 个正例。
   准确表述是「在该基准上未观察到收益」，应保留可切换开关并继续分析模型与数据的匹配度。
3. **单向量基线在本基准上很难被击败**：R1 在 Candidate Recall@20、Hit@5、Recall@5、MAP@20
   四项上都是最优。这个结果不好看，但如实保留。

完整表与自动归因见 [`eval/reports/comparison.md`](eval/reports/comparison.md)。上述是固定公开子集上的
描述性结果，不代表统计显著性，也不能外推为产品知识库效果。

### 5.2 P/Q runner 执行协议

先预热 60 个冻结 Planner 初始子问题。该命令只搜索 Web，不调用 LLM：

```bash
uv run python -m eval.prewarm_query_cache \
  --snapshot-id 2026-08-26-v1 \
  --max-queries 60 \
  --live
```

第一批 P1/P2 并发微基准共 10 个任务：

```bash
uv run python -m eval.run_orchestration \
  --snapshot-id 2026-08-26-v1 \
  --cache-mode replay-only \
  --groups P1 P2 \
  --rounds 1 \
  --max-tasks 10 \
  --output eval/reports/raw/parallel_p1_p2.jsonl \
  --live
```

Q2 的 Critic 可能产生未冻结的 revision query。先用 `record` 模式做非正式采集；该 raw 不能进入报告：

```bash
uv run python -m eval.run_orchestration \
  --snapshot-id 2026-08-26-v1 \
  --cache-mode record \
  --groups Q2 \
  --rounds 1 \
  --max-tasks 15 \
  --output eval/reports/raw/q2_cache_capture.jsonl \
  --live
```

缓存补齐后，Q1/Q2 正式两轮共 60 个任务，全部只读回放：

```bash
uv run python -m eval.run_orchestration \
  --snapshot-id 2026-08-26-v1 \
  --cache-mode replay-only \
  --groups Q1 Q2 \
  --rounds 2 \
  --max-tasks 60 \
  --output eval/reports/raw/quality_q1_q2.jsonl \
  --live
```

runner 每完成一题立即 `fsync` 追加 raw，进程中断后使用同一命令加 `--resume`；已完成的
`case_id + group + round` 会被跳过。snapshot、模型、并发、Critic 或固定子问题不一致时拒绝续跑。
`replay-only` 遇到 cache miss 会中止，不允许正式组临时联网补齐。

P/Q 正式 raw 产生后，与现有 R raw 合并生成报告：

```bash
uv run python -m eval.report \
  --retrieval-raw eval/reports/raw/retrieval_t2_100_local.jsonl \
  --task-raw eval/reports/raw/parallel_p1_p2.jsonl eval/reports/raw/quality_q1_q2.jsonl \
  --out eval/reports/comparison.md \
  --overwrite
```

当前只完成 runner 与离线测试，尚未执行上述付费 P/Q 任务，因此报告中仍只呈现 R 轨真实数字。

---

## 6. 报告格式

`eval/reports/comparison.md` 至少包含：

1. **实验元信息**：日期、模型、评测集版本、语料版本、各组配置快照
2. **R 轨检索对照表**

| 指标 | R1 向量 | R2 BM25 | R3 RRF | R4 +Rerank |
|------|---------|---------|--------|------------|
| Candidate Recall@20 | | | | |
| Hit@5 | | | | |
| MRR@5 | | | | |

3. **P/Q 编排对照表**

| 对照 | 对照组 | 实验组 | 独立变量 |
|------|--------|--------|----------|
| 并发微基准 | P1 | P2 | Researcher 并发上限 1 → 3 |
| Critic 增量 | Q1 | Q2 | Critic 关 → 开 |

报告对 P/Q 分别呈现引用有效性、关键点覆盖率、任务完成率、平均/P95 耗时、token、成本、
降级与反思触发率，并自动计算 P2-P1、Q2-Q1 的描述性差值。

4. **归因分析**：每项优化带来的变化及其解释
5. **负面结果**：变差的指标必须写出来（例如反思机制必然抬高 token 与耗时）
6. **威胁与局限**（见下节）

---

## 7. 结论的边界（面试必答）

以下局限必须写进报告，并在面试中主动说明。**主动承认局限比被问出来强得多。**

| 局限 | 说明 |
|------|------|
| R 轨子集 | 为控制本地建库时间，只抽取公开数据集的一部分 query 与候选，结论不等于完整 T2Ranking 排行 |
| P/Q 轨样本量小 | P 轨 5 题、Q 轨 15 题不足以得到统计显著性，结论仅为方向性参考 |
| 覆盖率指标弱 | 关键词匹配无法判断语义正确性 |
| P/Q 轨评测集自建 | 出题人与开发者是同一人，存在选择偏差；R 轨 qrels 不受此问题影响 |
| 联网证据边界 | query 缓存保证组间证据一致，但缓存只代表一次采集窗口 |
| 无人工标注质量分 | 报告优劣未经第三方评价 |

> **对应的面试回答思路**：承认样本量与偏差问题，说明控制变量做法（同一评测集、同窗口、多轮取均值），并指出如果继续做会引入 LLM-as-judge 双盲评分与更大样本量。这个回答比"我提升了 30%"有说服力得多。

---

## 8. 简历数字的取用规则

1. 只取**组间对照**的数字，不取绝对值（"从 X 提升到 Y"比"达到 Y"更可信）
2. 数字后必须能立刻说出**归因**（是哪一项优化带来的）
3. 负面代价要能一并说出（例如"引入反思后质量提升，但 token 上升 N%，因此设了 2 轮硬上限"）
4. 报告中不存在的数字，简历上一律不写
