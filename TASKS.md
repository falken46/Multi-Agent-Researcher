# Multi-Agent 研究助手 - 任务拆解清单

> 按 Phase 顺序执行，每个 Phase 完成后暂停验收。

## Phase 0: 项目初始化（预计 0.5 天）

- [x] T0.1 创建项目目录结构（按 ARCHITECTURE.md 4.目录结构）
- [x] T0.2 初始化 git 仓库
- [x] T0.3 创建 requirements.txt
- [x] T0.4 创建 .env.example（DEEPSEEK_API_KEY, TAVILY_API_KEY）
- [x] T0.5 创建 .gitignore
- [x] T0.6 编写初版 README.md

**验收标准**：`pip install -r requirements.txt` 成功

---

## Phase 1: 工具层实现（预计 1 天）

- [x] T1.1 实现 `tools/web_search.py`（Tavily 优先，DuckDuckGo 备选）
- [x] T1.2 实现 `tools/web_fetch.py`（requests + beautifulsoup4）
- [x] T1.3 编写 `tests/test_tools.py`，每个工具至少 2 个测试
- [x] T1.4 手动验证：在 Python REPL 中调用工具能返回真实数据

**验收标准**：两个工具能独立工作并返回结构化数据

---

## Phase 2: 状态定义与 Planner Agent（预计 1 天）

- [x] T2.1 实现 `agents/state.py`：定义 ResearchState TypedDict
- [x] T2.2 编写 `prompts/planner_system.md`（含 2 个 Few-shot 示例）
- [x] T2.3 实现 `agents/planner.py`：planner_node 函数
- [x] T2.4 编写 `tests/test_planner.py`：mock LLM，验证 JSON 解析
- [x] T2.5 手动测试：单独运行 planner_node，验证输出结构

**验收标准**：给定主题能稳定输出 3-5 个 JSON 格式的子问题

---

## Phase 3: Researcher Agent（预计 1.5 天）

- [x] T3.1 编写 `prompts/researcher_system.md`
- [x] T3.2 实现 `agents/researcher.py`：researcher_node 函数
- [x] T3.3 实现子问题循环 + 工具调用 + 资料摘要
- [x] T3.4 实现错误捕获与 errors 日志写入
- [x] T3.5 编写 `tests/test_researcher.py`：mock 工具调用
- [x] T3.6 手动测试：单独运行，输入 3 个子问题，观察输出

**验收标准**：每个子问题都能产出含来源 URL 的资料摘要

---

## Phase 4: Writer Agent（预计 0.5 天）

- [x] T4.1 编写 `prompts/writer_system.md`（含 Markdown 模板示例）
- [x] T4.2 实现 `agents/writer.py`：writer_node 函数
- [x] T4.3 实现 prompt 拼装（topic + questions + results）
- [x] T4.4 编写 `tests/test_writer.py`
- [x] T4.5 手动测试：用静态数据生成报告

**验收标准**：能输出结构完整的 Markdown 报告

---

## Phase 5: LangGraph 状态机集成（预计 1 天）

- [x] T5.1 实现 `agents/graph.py`：构建 StateGraph
- [x] T5.2 添加 planner → researcher → writer 主路径
- [x] T5.3 实现 `should_retry` 条件边逻辑
- [x] T5.4 添加 retry_count 上限保护（max=2）
- [x] T5.5 编写 `tests/test_graph.py`：端到端 mock 测试
- [x] T5.6 手动测试：完整运行一次"AI Agent 趋势"主题

**验收标准**：状态机能完整跑通三 Agent 流程，含失败重试

---

## Phase 6: FastAPI 后端（预计 1 天）

- [ ] T6.1 实现 `backend/api.py`：基础 FastAPI app
- [ ] T6.2 实现 `/research` POST 接口
- [ ] T6.3 实现 `backend/streaming.py`：SSE 流式推送
- [ ] T6.4 集成 LangGraph 的 stream 模式（按 node 推送状态）
- [ ] T6.5 添加 CORS 配置（允许 Streamlit 访问）
- [ ] T6.6 手动测试：用 curl 触发并观察 SSE 输出

**验收标准**：后端能流式返回每个 Agent 的状态变更

---

## Phase 7: Streamlit 前端（预计 1 天）

- [ ] T7.1 实现 `frontend/app.py`：基础布局
- [ ] T7.2 实现主题输入与启动按钮
- [ ] T7.3 实现 SSE 客户端订阅与进度展示
- [ ] T7.4 实现 Agent 状态分块展示（Planner / Researcher / Writer）
- [ ] T7.5 实现最终 Markdown 渲染与下载按钮
- [ ] T7.6 手动测试：浏览器端走完完整流程

**验收标准**：前端能稳定展示完整研究流程并下载报告

---

## Phase 8: Docker 容器化（预计 0.5 天）

- [ ] T8.1 编写 Dockerfile.backend
- [ ] T8.2 编写 Dockerfile.frontend
- [ ] T8.3 编写 docker-compose.yml
- [ ] T8.4 测试 `docker-compose up` 一键启动
- [ ] T8.5 更新 README.md，补充 Docker 启动说明

**验收标准**：在干净环境通过 docker-compose 启动并完成一次研究任务

---

## Phase 9: 端到端验收测试（预计 0.5 天）

- [ ] T9.1 编写 `tests/test_e2e.py`：mock 模式下的完整流程测试
- [ ] T9.2 运行 5 个真实主题（参考 TESTING.md）
- [ ] T9.3 撰写验收报告 `tests/acceptance_report.md`
- [ ] T9.4 录制一次完整 Demo 演示（截图或 GIF）

**验收标准**：5 个主题中至少 4 个能完整产出合格报告

---

## 总体时间预算

| 阶段 | 预计耗时 | 累计 |
|------|----------|------|
| Phase 0-1 | 1.5 天 | 1.5 天 |
| Phase 2-4 | 3 天 | 4.5 天 |
| Phase 5 | 1 天 | 5.5 天 |
| Phase 6-7 | 2 天 | 7.5 天 |
| Phase 8-9 | 1 天 | 8.5 天 |
| **合计** | **8.5 天** | （每天 4-6 小时投入） |

## 阶段间检查点

每个 Phase 完成后，作者必须：
1. ✅ 所有任务勾选完毕
2. ✅ 对应测试通过
3. ✅ 能用自己的话讲清楚本阶段实现了什么
4. ✅ Git commit（commit message 标注 Phase 名）
