# Phase 13 评测数据

本目录只追踪评测集说明与人工维护的小型编排题集。公开检索数据的原始文件和
转换产物体积较大，由命令生成并通过 `.gitignore` 排除。

## R 轨：T2Reranking 共享语料池

检索消融使用中文公开数据集 `C-MTEB/T2Reranking`。它源自
`THUIR/T2Ranking`；原项目声明 Apache-2.0 许可证。正式引用数据时应同时引用
T2Ranking 与 MTEB/C-MTEB。

准备命令：

```bash
uv run python -m eval.prepare_retrieval_dataset --queries 100 --seed 42
```

命令会下载约 120 MB 的 Parquet 到 `eval/.cache/t2_reranking/`，然后在
`eval/dataset/t2_reranking/` 生成：

- `corpus.jsonl`：所有抽中查询共用的 passage 池；
- `queries.jsonl`：查询、gold passage 与原数据集候选关系；
- `qrels.jsonl`：查询与相关 passage 的结构化标注；
- `metadata.json`：数据来源、抽样参数和实际规模。

所有 query 必须从整个共享语料池检索，不能只在自身的
`source_candidate_passage_ids` 中排序。后者只保留用于审计原数据结构。

T2Ranking 已经把文本定义为 passage 检索单元，因此转换器不会再按产品知识库
的 `CHUNK_SIZE` 二次切分；每个 passage 直接适配为一个 `Chunk`，其 ID 与 qrels
一致。

如需重新抽样，显式传入 `--overwrite`。本流程不做额外的语料 hash 验证，
只把数据来源、随机种子和抽样上限写入 `metadata.json`。

## P/Q 轨：端到端编排题集

`orchestration_qa.jsonl` 保存本地、联网和混合任务。它与 R 轨分开：R 轨只测
向量、BM25、RRF 与 rerank；P/Q 轨测任务完成率、覆盖、引用、trace 成本与耗时。

编排实现进一步拆成 P/Q 两个对照：P1/P2 使用 5 道 KB 题，只改变 Researcher 并发上限；
Q1/Q2 使用全部 15 题，只改变 Critic 是否启用。四组都使用文件内冻结的 `sub_questions`，
不调用 Planner LLM。

首次正式运行前，先显式预热初始 Web query：

```bash
uv run python -m eval.prewarm_query_cache \
  --snapshot-id 2026-08-26-v1 --max-queries 60 --live
```

runner 必须显式传 `--live` 与 `--max-tasks`。正式组统一使用 `--cache-mode replay-only`；
每题完成后立即追加 raw，可用 `--resume` 跳过身份完全一致的已完成任务。完整 P/Q 命令见
项目根目录 `EVAL.md`。

## 运行 R1—R4

公开子集准备完成后，运行本地检索消融：

```bash
uv run python -m eval.run_retrieval --groups R1 R2 R3 R4
```

runner 只允许本地 `fastembed` 或测试用 `fake` embedding；R4 固定使用 ONNX
reranker，不调用 LLM、Web 或 fallback。评测索引写入 `eval/.cache/`，结构化候选、
Top5、gold 与分数证据写入 `eval/reports/raw/`，两者均不进入 Git。

从指定 raw 文件重新计算指标并生成 Markdown 报告：

```bash
uv run python -m eval.report \
  --retrieval-raw eval/reports/raw/retrieval_t2_100_local.jsonl \
  --out eval/reports/comparison.md \
  --overwrite
```

不要把 `--limit` smoke 结果写进项目介绍；正式 R 轨使用全部 100 个 query。

## 来源

- C-MTEB T2Reranking: https://huggingface.co/datasets/C-MTEB/T2Reranking
- T2Ranking: https://github.com/THUIR/T2Ranking
