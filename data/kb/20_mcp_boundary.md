# MCP 工具边界

MCP Server 可以把 deep_research 与 kb_search 暴露给外部客户端。MCP 负责协议适配和参数校验，核心业务仍复用已有图与检索工具。这样同一能力既能被 Web 前端调用，也能被支持 MCP 的开发工具调用，而不会复制一份 Agent 逻辑。
