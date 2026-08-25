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
| generated_at_utc | 2026-08-25T21:21:50+00:00 |
| groups | R1, R2, R3, R4 |
| observation_count | 400 |
| query_count | 100 |
| raw_files | eval/reports/raw/retrieval_t2_100_local.jsonl |
| rerank_backend | onnx |
| rerank_model | BAAI/bge-reranker-base |
| rrf_k | 60 |

## R 轨：检索质量

| 组 | 题目数 | 观测数 | Candidate Recall@20 | Hit@5 | MRR@5 | Recall@5 | nDCG@5 | MAP@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | 100 | 100 | 93.64% | 96.00% | 0.7238 | 45.46% | 0.6254 | 0.6204 |
| R2 | 100 | 100 | 85.65% | 94.00% | 0.7575 | 40.81% | 0.5961 | 0.5538 |
| R3 | 100 | 100 | 92.27% | 95.00% | 0.7777 | 43.89% | 0.6359 | 0.6144 |
| R4 | 100 | 100 | 92.27% | 93.00% | 0.7377 | 44.27% | 0.6255 | 0.6140 |

## R 轨归因分析

- R3 相较 R1：Candidate Recall@20 -1.37 个百分点，Hit@5 -1.00 个百分点，MRR@5 +0.0538，Recall@5 -1.56 个百分点，nDCG@5 +0.0105，MAP@20 -0.0059。
- R3 相较 R2：Candidate Recall@20 +6.62 个百分点，Hit@5 +1.00 个百分点，MRR@5 +0.0202，Recall@5 +3.08 个百分点，nDCG@5 +0.0398，MAP@20 +0.0606。
- R4 相较 R3：Candidate Recall@20 +0.00 个百分点，Hit@5 -2.00 个百分点，MRR@5 -0.0400，Recall@5 +0.38 个百分点，nDCG@5 -0.0104，MAP@20 -0.0004。
- R4 只重排 R3 的同一候选集，因此 Candidate Recall@20 必然不变；重排效果应主要看 Recall@5、nDCG@5 与 MAP@20，而非首命中型的 Hit@5。
- 上述差值是当前固定子集上的描述性结果，不代表统计显著性；负向结果同样保留。

## 指标边界

- 该检索集平均每个 query 含多个正例，Hit@5 与 MRR@5 只反映「首个正例的位置」，会较早饱和；Recall@5、nDCG@5 与 MAP@20 才度量「多个正例整体排得好不好」，也是源基准的官方口径。
- MAP@20 的 AP 以 min(正例数, 20) 归一化，使正例数超过截断深度的 query 仍可取到 1.0，避免组间比较被各 query 的正例数量分布带偏。
- nDCG@5 使用二值增益：冻结的 qrels 只标注相关与否，没有分级判断。
- Coverage 是 Unicode/空白归一化后的关键词或同义短语覆盖率，不代表语义正确性。
- Citation validity 只验证报告中的数字引用编号是否存在于输入引用集合，不证明来源支持对应结论。
- token、成本、调用次数、fallback、revision 与耗时均来自结构化 trace summary，不从普通日志提取。
- Critic 自身评分不作为 Critic 有效性的独立质量证据。
