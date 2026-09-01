# SSE 流式协议与前端进度

## SSE 适合单向持续更新

研究任务由客户端提交一次主题，服务端随后持续发送节点进度、降级、返工和最终报告。主要数据方向是服务端到浏览器，因此 Server-Sent Events 比双向 WebSocket 更简单。若场景需要高频双向协作或二进制帧，再考虑 WebSocket。

## HTTP 响应保持事件流

后端返回 text/event-stream，并在连接存活期间逐个写出事件。每个事件使用稳定类型和 JSON data，前端按 event 分派处理。不要让前端解析“正在运行 Critic……”这样的自由文本来判断状态，文案变化会破坏逻辑。

## start 事件交付身份

流开始后应尽早发送 start，其中包含 thread_id 和 trace_id。客户端保存 thread_id 供恢复请求使用，trace_id 用于定位本次执行的 JSONL。若等到任务结束才返回标识，中途断连的用户就无法可靠恢复或排查。

## progress 描述节点更新

LangGraph updates 流给出节点局部状态，后端将其转换为 progress 事件。前端可以展示 Planner、Researcher、Critic 和 Writer 的当前阶段，但不应把完整内部状态原样发送，以免泄露长证据、敏感信息或造成巨大流量。

## custom 事件表达业务动作

fallback、revision、critic_start 和 critic_done 等事件不是普通节点完成，它们解释图内重要行为。Agent 通过 StreamWriter 发出结构化 payload，后端转成 SSE。这样 UI 能即时展示联网降级和定向返工，而不等待节点结束。

## usage 必须来自 trace

任务完成后，后端调用 summarize(trace_id)，再发送 usage 事件，包括 token、成本、耗时、调用数和行为计数。前端只负责展示，不能通过收到多少条 progress 或字符串时间戳自行计算指标。

## complete 与 error 是终态

complete 携带最终报告和必要状态摘要，error 携带可解释的失败信息。发送终态后流应正常结束并释放 saver 连接。某个子问题失败但 Writer 仍成功时，可以在 complete 中包含 errors，而不必把整次任务标为传输错误。

## 心跳与代理超时

长时间没有事件时，反向代理或浏览器可能认为连接空闲。生产部署可以发送注释心跳，并配置代理禁用不合适的缓冲。个人本地演示节点事件较频繁，但文档应区分应用逻辑正确与公网基础设施可靠。

## 重连不等于任务恢复

SSE 客户端的网络重连只能重新建立数据连接，不能自动推断 LangGraph 从哪里继续。任务恢复依赖稳定 thread_id 和 checkpoint。协议层的 Last-Event-ID 可用于事件重放设计，但若服务端没有保存可重放事件，就不能声称断线消息一定补齐。

## 断连时清理异步资源

客户端关闭页面后，服务端应感知取消，结束不再需要的生成器并关闭 SQLite 连接。若后台任务选择继续运行，也需要明确队列和结果查询机制。含糊地让协程悬挂会持续消耗模型额度并占用连接。

## Streamlit 消费事件

前端发送 POST 请求后逐行解析 SSE，依据 event 更新四个节点状态、Critic 评分、fallback 提示和 usage 面板。未知事件应忽略或记录，而不是让页面崩溃，这为后端以后增加事件类型保留兼容空间。

## 流式测试不需要真实模型

后端测试可以替换图的异步流，依次产生 updates 和 custom 数据，断言 SSE 顺序、事件名称和终态。还应覆盖异常转 error、恢复参数校验和连接关闭。真实浏览器演示属于更高层验证，不应成为基础 CI 前提。
