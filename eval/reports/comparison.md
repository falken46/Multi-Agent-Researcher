# DeepResearch Agent 评测报告

> 本报告仅呈现结构化评测原始记录与 trace 汇总计算得到的指标；
> 缺少结构化证据的实验会整节省略，不生成空白值或占位值。

## 实验元信息

| 字段 | 值 |
|---|---|
| candidate_k | 20 |
| dataset | C-MTEB/T2Reranking |
| embedding_backend | fastembed |
| embedding_model | BAAI/bge-small-zh-v1.5 |
| final_k | 5 |
| generated_at_utc | 2026-08-25T20:17:02+00:00 |
| groups | R1, R2, R3, R4 |
| observation_count | 400 |
| query_count | 100 |
| raw_files | eval/reports/raw/retrieval_t2_100_local.jsonl |
| rerank_backend | onnx |
| rerank_model | BAAI/bge-reranker-base |
| rrf_k | 60 |

## R 轨：检索质量

| 组 | 题目数 | 观测数 | Candidate Recall@20 | Hit@5 | MRR@5 |
|---|---:|---:|---:|---:|---:|
| R1 | 100 | 100 | 93.64% | 96.00% | 0.7238 |
| R2 | 100 | 100 | 85.65% | 94.00% | 0.7575 |
| R3 | 100 | 100 | 92.27% | 95.00% | 0.7777 |
| R4 | 100 | 100 | 92.27% | 93.00% | 0.7377 |

## R 轨归因分析

- R3 相较 R1：Candidate Recall@20 -1.37 个百分点，Hit@5 -1.00 个百分点，MRR@5 +0.0538。
- R3 相较 R2：Candidate Recall@20 +6.62 个百分点，Hit@5 +1.00 个百分点，MRR@5 +0.0202。
- R4 相较 R3：Candidate Recall@20 +0.00 个百分点，Hit@5 -2.00 个百分点，MRR@5 -0.0400。
- 上述差值是当前固定子集上的描述性结果，不代表统计显著性；负向结果同样保留。

## 指标边界

- Coverage 是 Unicode/空白归一化后的关键词或同义短语覆盖率，不代表语义正确性。
- Citation validity 只验证报告中的数字引用编号是否存在于输入引用集合，不证明来源支持对应结论。
- token、成本、调用次数、fallback、revision 与耗时均来自结构化 trace summary，不从普通日志提取。
- Critic 自身评分不作为 Critic 有效性的独立质量证据。
