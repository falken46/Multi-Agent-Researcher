# TypedDict 共享状态

ResearchState 使用 TypedDict 明确字段，例如 topic、sub_questions、research_results 和 errors。状态 schema 是 Agent 之间的协议：Planner 只写规划结果，Researcher 消费子问题，Writer 消费研究材料。显式类型可以减少字段拼写错误，也让图的输入输出更容易理解。
