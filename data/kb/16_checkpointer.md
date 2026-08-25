# Checkpointer 断点续跑

LangGraph Checkpointer 保存每一步的共享状态。任务因进程退出或人工审批暂停后，可以使用 thread_id 找回状态并继续，而不是从头重复昂贵检索。SQLite 适合单机演示，生产系统可根据并发和运维需求替换持久化后端。
