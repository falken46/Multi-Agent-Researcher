# Cross-encoder 重排

初次召回追求覆盖率，rerank 再对 query 与候选正文做成对判断，提升前几名的精度。项目可选择 FastEmbed ONNX cross-encoder、LLM 打分或关闭重排。重排只处理有限候选，避免对整个知识库执行昂贵模型推理。
