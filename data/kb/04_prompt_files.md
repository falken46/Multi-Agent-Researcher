# Prompt 文件化

系统提示词放在 prompts 目录的 Markdown 文件中，由 load_prompt 按名称加载。Prompt 与 Python 逻辑分离后，修改角色约束不会制造大量代码 diff，面试时也能清楚说明“行为配置”和“执行代码”的边界。空文件或缺失文件应尽早报出明确错误。
