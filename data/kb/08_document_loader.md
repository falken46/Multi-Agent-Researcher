# 知识库文档加载

loader 按扩展名处理 Markdown、TXT 和 PDF。文本文件直接读取，PDF 使用 pypdf 按页提取文字，并把 source_path、file_type、page 写入 metadata。某个损坏文件只进入失败清单，不应让整批建库任务中断。
