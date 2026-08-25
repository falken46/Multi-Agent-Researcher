# Chroma 向量库

Chroma 在本地目录持久化 chunk 的向量、正文和 metadata。项目封装 add、query、count 三个操作，业务层不感知集合创建和距离字段等实现细节。建库和查询必须使用同一个 embedding 模型与维度，否则向量不可比较。
