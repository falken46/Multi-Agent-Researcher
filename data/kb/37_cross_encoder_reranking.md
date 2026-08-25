# Cross-encoder 精排与候选预算

## 召回和重排目标不同

初次召回追求不要遗漏相关内容，通常快速取较大的候选集合；rerank 追求把真正回答 query 的文本排到前几位。把两者混成一步，会迫使昂贵模型扫描整个知识库，或让轻量召回承担它不擅长的细粒度判断。

## Cross-encoder 联合编码

Cross-encoder 同时读取 query 与候选正文，直接预测这对文本的相关性。它能识别共享大量术语但回答方向不同的 hard negative，通常比独立编码后算余弦更精细。代价是每个候选都要单独推理，无法像向量那样预计算文档表示。

## 候选数量限制成本

pipeline 先让向量和 BM25 产生有限候选并经 RRF 去重，再只对这些候选重排。retrieval_top_k 控制每路召回宽度，rerank_top_n 控制最终返回数量。若前一阶段没召回 gold，重排再强也无法找回，因此需要同时报告召回与排序指标。

## ONNX 本地重排

FastEmbed TextCrossEncoder 可在本地加载 ONNX 模型，对 query 和候选正文列表给分。首次运行可能下载模型，CI 不应依赖这一过程。模型返回分数的范围取决于实现，不能未经验证就当成标准概率。

## LLM 重排的结构化分数

LLM reranker 把 chunk_id 与候选正文发送到统一 LLM 入口，请求返回每个 ID 的零到一分数。它更灵活但消耗 token、增加延迟，也可能漏项或输出非法 JSON。所有模型调用仍必须带 node 和 trace_id，以保持成本完整。

## 解析失败使用中性值

LLM 分数解析失败时，可为缺失候选提供中性分数并按 chunk_id 稳定排序，使主流程继续。降级不能假装模型成功，应记录 warning 和 trace。中性值会削弱重排效果，评测报告需要保留这类失败样本。

## none 后端保留融合顺序

关闭重排时，rerank 函数直接截取 RRF 排名的前 top_n。这个配置是测量重排增益的必要基线。若关闭后又使用另一套排序规则，对照组就不再只差一个变量。

## 分数覆盖改变阈值语义

经过 cross-encoder 后，RetrievalResult.score 被替换为重排模型分数，channel 增加 rerank 标记。未重排时 score 是 RRF 倒数排名和。两者量纲不同，因此“最高分低于某阈值”必须知道当前 backend，不能跨阶段共用想当然的数值。

## MRR 对重排更敏感

如果 gold 在重排前第五、重排后第一，Hit@5 都是命中，但倒数排名从 0.2 变成 1，MRR 清楚反映顶部改善。因此报告应同时给 Hit@5 和 MRR。只看 Hit@K 可能低估重排，也不能解释用户首先看到的结果是否更准。

## Hard negative 是关键样本

Cross-encoder 的优势通常出现在候选都谈论相同主题、但只有一段满足问题关系时。随机无关文本在召回阶段已经很容易排除，无法检验精排。数据集需要显式标注与 gold 相近但缺少核心答案的干扰 chunk。

## 重排失败应退回融合结果

模型加载、推理或 LLM 调用失败时，pipeline 记录 rerank channel error，并返回 RRF 的前 top_n，而不是让整个知识库不可用。该行为保持可用性，但结果 channel 和 trace 必须让使用者知道精排未生效。

## 对照控制模型和候选

评估 rerank 时，前置语料、embedding、BM25、RRF k、retrieval_top_k 与 query 必须相同，只切换 rerank_backend。若同时扩大候选池或更换 embedding，MRR 变化无法独立归因给 cross-encoder。
