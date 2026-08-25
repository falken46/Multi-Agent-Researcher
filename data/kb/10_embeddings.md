# Embedding 后端

Embedding 把文本映射成稠密向量，使语义相近的查询和段落在向量空间里更接近。项目支持 fastembed 本地 ONNX、OpenAI 兼容远程 HTTP 接口和 fake 测试后端。fake 后端使用确定性哈希，不下载模型，保证 CI 离线可复现。
