# 集中配置

core.config 用 Pydantic Settings 声明环境变量、类型、默认值和跨字段校验。业务代码不直接调用 os.getenv。新增配置时同步更新配置类、.env.example 与 TECH_STACK，能避免“代码支持了参数，但使用者不知道如何设置”的文档漂移问题。
