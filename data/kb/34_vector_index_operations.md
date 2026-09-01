# Chroma 向量索引的窄接口

## 封装只暴露三个动作

ChromaVectorStore 对上层只提供 add、query 和 count。集合创建、持久化客户端、距离字段和 metadata 清洗留在适配器内部。窄接口减少业务代码对 Chroma API 的耦合，未来更换向量库时，pipeline 不需要理解供应商细节。

## 集合名称隔离用途

chroma_collection 由集中配置提供，测试使用独立名称和临时目录，避免覆盖开发索引。集合名为空应在 Settings 校验阶段失败。不同 embedding 模型即使处理同一语料，也不应无声明地复用同一集合。

## reset 用于确定性重建

建库时 reset=True 会删除已有同名集合再创建，保证删除的语料不会残留。仅 upsert 当前 chunk 无法清理本轮已不存在的旧 ID。重建是简单可靠的演示策略，大规模生产索引则可能需要版本化集合和原子切换。

## upsert 依赖稳定 chunk_id

add 同时写入 chunk.id、正文、metadata 和预计算 embedding。ID 稳定时重复构建不会产生重复记录。写入前必须检查 chunks 与 embeddings 数量一致，不能让 zip 静默截断，否则部分来源会在无错误提示的情况下消失。

## Metadata 需要基础类型

Chroma metadata 接受的值类型有限，适配器保留字符串、数字和布尔值，其他非空对象转换为字符串。复杂嵌套结构不应依赖这种转换保存，因为之后无法可靠恢复类型。核心定位字段应始终使用简单类型。

## cosine 距离转相似度

查询返回距离，适配器将其转换为截断在零到一之间的 1-distance 分数。这个值便于展示，却依赖所选空间和 Chroma 语义。它不是跨排序阶段统一置信度，不能直接与 RRF 或 cross-encoder 分数共用同一个未经校准阈值。

## top_k 不能超过集合数量

query 使用 n_results=min(top_k, count)，避免小集合请求过多候选。如果集合只有 20 个 chunk 且 top_k 也是 20，每次向量查询都会返回全部语料，候选召回指标失去区分度。评测语料规模应显著大于 top_k。

## 查询结果恢复统一模型

适配器把 Chroma 返回转换为 RetrievalResult，包括 chunk_id、text、source、chunk_index、score、channel 和 metadata。后续 BM25、RRF 与 reranker 只处理统一结构，不需要判断结果来自何种数据库。

## 空集合返回空列表

当 top_k 非正或集合为空时，query 直接返回空结果。pipeline 再判断另一通道是否可用。向量索引缺失和“查询确实没有相关内容”不是同一个事件，异常路径应记录 channel_errors，正常空结果则保留为无候选。

## 持久化目录是运行数据

Chroma 文件位于配置目录，适合容器挂载和本地复用，但不应提交 Git。索引可由受版本控制的语料和配置重建。评测报告应直接记录语料版本、chunk 数和 splitter 参数，不需要为此增加额外完整性摘要。

## 向量通道的确定性边界

给定相同向量和集合，排序通常稳定；模型版本、硬件数值差异或近似索引参数可能影响相近候选顺序。使用 chunk_id 作为相同分数时的二级排序能增强可复现性，但真实模型评测仍应记录环境。

## 测试使用临时持久化目录

测试构造 fake embedding、独立集合和 runtime_dir，建库后断言 count 与 chunk 数一致，再查询已知主题。结束后删除运行目录。这样既覆盖真实 Chroma 读写，又不污染开发者的 data/chroma。
