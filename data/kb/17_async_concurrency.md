# 异步并发研究

多个独立子问题可以用 asyncio 并发研究，但必须用 Semaphore 限制同时请求数量，并为外部调用设置超时。asyncio.gather 使用 return_exceptions=True 后，单个子问题失败不会取消全部任务，节点可以把失败写入 errors 并保留其他成功结果。
