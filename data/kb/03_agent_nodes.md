# Agent 节点边界

Planner、Researcher、Critic、Writer 分别放在独立文件，每个文件只暴露一个 node 函数。节点之间不直接调用，而是由 LangGraph 根据边来编排。这样可以单独测试每个角色，也避免一个巨型函数同时承担规划、检索、评价和写作。
