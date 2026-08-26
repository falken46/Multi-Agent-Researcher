# GitHub 发布清单

> 本清单供作者自行执行。Codex 不会代替作者提交、改远端仓库或推送。

## 1. 推送前安全检查

1. 先轮换曾在终端输出中暴露过的 DeepSeek 与 Tavily Key，并只把新 Key 放进本地 `.env`。
2. 确认 `.env` 被忽略且没有被 Git 跟踪：

```powershell
git check-ignore -v .env
git ls-files --error-unmatch .env
```

第二条命令预期失败；如果它返回了 `.env`，先停止发布并将该文件从 Git 索引移除。

3. 在已跟踪内容和暂存区中检查疑似真实密钥：

```powershell
git grep -n -I -E "sk-[A-Za-z0-9]{20,}|tvly-[A-Za-z0-9_-]{20,}"
git diff --cached
```

第一条命令预期无输出；第二条要人工确认没有 `.env`、运行时 trace、模型缓存或个人路径。

## 2. 发布前回归

```powershell
uv sync --frozen --group dev
uv run --frozen ruff check .
uv run --frozen python -m pytest -m "not live"
docker compose config --quiet
```

需要重新验证完整容器启动链时，再运行：

```powershell
docker compose up --build
```

验收口径：indexer 成功后 backend 启动，`http://127.0.0.1:8000/health` 返回健康状态，随后 frontend 可在 `http://127.0.0.1:8501` 打开。不要为这一步填写真实 LLM 效果或成本数字。

## 3. 本地提交

当前开发分支是 `feat/deepresearch-upgrade`。先用 `git status --short` 和 `git diff --stat` 检查范围，再由作者自行分批暂存和提交。

Phase 16 建议提交消息：

```text
docs: 补齐简历映射、面试口述稿与发布清单
```

如果 Phase 15 尚未单独提交，建议先把 Docker、Compose、CI 及其测试作为独立提交，再提交 Phase 16 文档，便于面试时按阶段展示工程演进。

## 4. 重命名仓库并推送个人账号

推荐在 GitHub 仓库的 **Settings → General → Repository name** 中把仓库改为：

```text
deepresearch-agent
```

重命名后，把本地 `origin` 更新到个人账号的 SSH Host 别名；该别名已用于当前远端，可避免误用学校账号：

```powershell
git remote set-url origin git@github-personal:falken46/deepresearch-agent.git
git remote -v
git push -u origin feat/deepresearch-upgrade
```

然后在 GitHub 上从 `feat/deepresearch-upgrade` 向默认分支创建 Pull Request。合并前确认 Actions 中 `Ruff and pytest` job 真实绿色；只有这一步完成后，README 才能添加绿色 CI 徽章。

## 5. 仓库展示信息

建议 Description：

```text
LangGraph multi-agent research assistant with hybrid RAG, evaluation, tracing, MCP, Docker Compose and offline CI.
```

建议 Topics：

```text
langgraph  multi-agent  rag  retrieval  mcp  fastapi  streamlit  docker  evaluation
```

建议 About 区域保留 Issues，关闭不使用的 Packages 与 Releases；如果没有稳定公网部署地址，不填写 Website，避免产生失效入口。

## 6. 推送后的最终验收

- [ ] 仓库名为 `deepresearch-agent`，Description 与 Topics 已填写。
- [ ] feature 分支已推送，Pull Request 的 CI 真实绿色。
- [ ] README 中所有相对链接可以从 GitHub 正常打开。
- [ ] 仓库历史和 Actions 日志中没有真实 API Key。
- [ ] 项目首页能快速看到架构、启动方式、真实检索结论、边界和面试材料入口。
- [ ] 完成真实 UI 截图后再添加 Demo 图片；图片必须标注运行模式，不把固定事件演示当成模型效果。
