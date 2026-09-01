# Trace 事件模型与指标事实源

## trace_id 串联一次任务

任务入口生成唯一 trace_id，随后 Planner、Researcher、Critic、Writer、工具与 LLM 网关都复用它。按这个标识读取 JSONL，可以还原一次任务的路径和资源消耗。thread_id 用于 checkpoint 恢复，生命周期可能跨多次执行，不能替代 trace_id。

## JSONL 采用单行事件

每一行是独立 JSON 对象，包含 trace_id、时间戳、event、node 和 payload。追加写入时进程意外中断，通常只会损坏最后一行；读取器可以跳过坏行并继续处理前面的完整事件。普通 JSON 数组则需要重写结尾，长任务中更脆弱。

## node_start 与 node_end

节点入口记录输入数量、模式和关键配置，出口记录状态、输出数量与 latency_ms。成对事件适合展示执行路径，但节点耗时不能简单相加得到并行任务的墙钟时间。端到端耗时应优先由 task_start 与 task_end 的时间差计算。

## llm_call 记录资源消耗

成功模型调用写入模型名、prompt_tokens、completion_tokens、latency_ms、cost 和 attempt。失败尝试应标记 success=false 或发出 error，避免把没有响应的 token 当真实值。按 node 聚合后可以定位哪个角色占据主要成本。

## retrieval 描述检索行为

检索事件至少记录查询摘要、可用通道、命中数量、通道错误、最高分和耗时。若要计算 Hit@K 与 MRR，还必须在评测原始产物或 trace 中保存有序 chunk_id；只有 max_score 无法知道 gold 排在第几名。

## fallback 解释为什么联网

本地结果低于阈值或知识库通道失败时，Researcher 发出 fallback 事件，payload 包含查询、实际最高分、阈值和原因。只记录“调用了 web_search”无法判断这是设计内降级还是误触发，也无法校准阈值。

## revision 区分主动返工

Critic 导致的质量返工应记录轮次、返工前评分和 missing_aspects。技术重试则使用单独事件或 attempt 字段。两个行为都可能增加调用数，但一个反映可靠性问题，一个反映质量策略，评测中必须分开统计。

## error 需要结构化字段

error 事件使用 node、异常类型和消息，而不是只写一段日志文本。评测脚本可以按类型计数，开发者也能定位失败边界。消息需要截断并去除敏感信息，异常对象本身不可直接序列化到 JSON。

## summarize 只聚合事件

summarize(trace_id) 逐行读取事件，计算 token、成本、LLM 次数、降级次数、返工次数和错误列表。它不解析自由文本日志，也不依赖前端显示状态。相同 trace 文件应始终得到相同汇总，这是指标可复现的基础。

## 并发写入需要锁

Researcher 多个线程可能同时向同一 trace 文件追加事件。进程内按路径加锁可避免两行字节交错，打开文件时固定 UTF-8 和换行符。该方案适合单进程演示；多进程共享目录时还需文件锁或集中式观测后端。

## Trace、日志与 SSE

日志服务于开发排查，可以自由描述；SSE 服务于当前连接的实时反馈；trace 是长期机器可读事实源。三者可能描述同一动作，但不能从 SSE 文案或日志正则计算指标。前端展示 usage 也应来自 trace 汇总。

## 评测原始结果保留溯源

批量评测除了 trace_id，还应记录样本 ID、实验组、轮次、数据集版本、语料版本和配置快照。报告只聚合版本与配置相同的记录。若把不同模型或不同语料的结果混在一张均值表中，再精确的小数也没有解释价值。
