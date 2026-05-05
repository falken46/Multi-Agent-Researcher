# Multi-Agent 研究助手 - 测试与验收标准

## 1. 测试分层

### 1.1 单元测试（pytest）
位于 `tests/test_*.py`，覆盖每个 Agent 与工具。

### 1.2 集成测试
`tests/test_graph.py` 测试完整 LangGraph 流程（mock LLM）。

### 1.3 端到端测试
真实调用 LLM + Web Search，跑 5 个验收主题。

## 2. 单元测试要求

### 2.1 工具测试

```python
# tests/test_tools.py
def test_web_search_returns_results():
    results = web_search("LangGraph tutorial", max_results=3)
    assert len(results) > 0
    assert all("url" in r for r in results)

def test_web_fetch_extracts_text():
    text = web_fetch("https://example.com")
    assert isinstance(text, str)
    assert len(text) > 0
```

### 2.2 Agent 测试（Mock LLM）

```python
# tests/test_planner.py
def test_planner_outputs_json_list(mocker):
    mock_response = create_mock_response('["Q1", "Q2", "Q3"]')
    mocker.patch("agents.planner.client.messages.create", return_value=mock_response)
    
    state = {"topic": "AI 趋势", "sub_questions": [], ...}
    result = planner_node(state)
    
    assert len(result["sub_questions"]) == 3
    assert all(isinstance(q, str) for q in result["sub_questions"])
```

### 2.3 状态机测试

```python
# tests/test_graph.py
def test_full_graph_happy_path(mocker):
    # mock 三个节点的 LLM 调用
    ...
    
    initial_state = {"topic": "测试主题", ...}
    final_state = graph.invoke(initial_state)
    
    assert final_state["sub_questions"]
    assert final_state["research_results"]
    assert final_state["final_report"]
```

## 3. 端到端验收主题集

以下 5 个主题用于真实 LLM + Web Search 验收：

### 主题 1：技术趋势类
**输入**：`2025 年 AI Agent 领域的主要技术趋势`

**预期产出**：
- 3-5 个子问题（覆盖架构、应用、挑战等维度）
- 每个子问题有真实 Web 来源
- 报告含摘要 + 各小节 + 参考来源

**通过标准**：报告结构完整，无明显事实错误。

---

### 主题 2：工具对比类
**输入**：`LangGraph vs LangChain：核心差异与适用场景`

**预期产出**：
- 子问题应包含两者的设计哲学、API 差异、迁移路径
- 报告应有对比表格（Markdown table）

**通过标准**：能产出对比表格且信息准确。

---

### 主题 3：人物/公司研究
**输入**：`Anthropic 公司 2025 年的产品矩阵`

**预期产出**：
- 涵盖 Claude 模型、API、Claude Code 等产品
- 每个产品有简短介绍

**通过标准**：至少覆盖 3 个真实产品线。

---

### 主题 4：行业分析
**输入**：`大模型推理优化的最新方法`

**预期产出**：
- 涉及 KV Cache、Speculative Decoding、量化等方向
- 含具体技术名词与简介

**通过标准**：至少识别 3 个有效优化方向。

---

### 主题 5：流程类
**输入**：`如何从 0 开始学习 Multi-Agent 系统开发`

**预期产出**：
- 子问题应包含基础知识、框架选择、实战路径
- 报告应有阶段性学习路线

**通过标准**：能输出结构化学习路径。

---

## 4. 验收报告模板

完成 5 个主题测试后，在 `tests/acceptance_report.md` 中按以下格式记录：

```markdown
# 验收报告

测试日期：YYYY-MM-DD
模型：deepseek-v4-pro / deepseek-v4-flash
搜索引擎：Tavily

## 主题结果

| 主题编号 | 主题 | 子问题数 | 检索成功率 | 报告质量 | 总评 |
|---------|------|---------|-----------|---------|------|
| 1 | AI Agent 趋势 | 5 | 5/5 | ✅ 优 | Pass |
| 2 | LangGraph vs LangChain | 4 | 4/4 | ✅ 良 | Pass |
| 3 | Anthropic 产品矩阵 | 5 | 4/5 | ⚠️ 中 | Partial |
| 4 | 推理优化 | 5 | 5/5 | ✅ 优 | Pass |
| 5 | Multi-Agent 学习 | 4 | 4/4 | ✅ 良 | Pass |

## 通过率：4/5（80%）

## 各 Agent 表现

### Planner
- ✅ JSON 格式输出稳定
- ⚠️ 主题 3 拆解粒度偏粗，建议优化 Prompt

### Researcher
- ✅ 检索资料相关性高
- ⚠️ 偶发 Tavily API 超时，已通过重试机制处理

### Writer
- ✅ Markdown 结构完整
- ⚠️ 主题 3 报告内容偏短，可能受 Researcher 输出限制

## LangGraph 状态机表现
- 重试机制触发：2 次（主题 3、主题 4）
- 全部成功完成最终报告

## 工程层面
- Docker 启动：✅ 一次成功
- SSE 流式推送：✅ 前端能实时看到进度
- 平均任务耗时：约 90 秒
```

## 5. 性能基准（参考）

| 指标 | 目标值 |
|------|--------|
| 单次完整研究任务 | < 3 分钟 |
| Planner 单步耗时 | < 10 秒 |
| Researcher 单子问题 | < 30 秒 |
| Writer 报告生成 | < 60 秒 |
| 后端启动时间 | < 5 秒 |
| Docker 镜像大小 | < 500 MB |

## 6. Demo 录制建议

最终验收时录制一段 1-2 分钟 Demo，建议内容：

1. 启动服务（`docker-compose up`）
2. 打开 Streamlit 前端
3. 输入主题（建议用主题 2"LangGraph vs LangChain"）
4. 展示 3 个 Agent 实时进度
5. 展示最终 Markdown 报告
6. 点击下载按钮

录制工具推荐：
- macOS：QuickTime / Kap
- Windows：ScreenToGif / OBS
- 跨平台：LICEcap（GIF 输出）
