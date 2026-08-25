# 统一 LLM 入口

core.llm 的 chat 与 achat 是项目唯一的模型调用入口。这里统一创建客户端、处理超时重试、收集 token、估算成本并写入 trace。Agent 只关心 messages、node 和 trace_id，不再各自实现重复的 SDK 调用代码，因此成本统计不会因为某个节点绕过网关而缺失。
