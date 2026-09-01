# 文档摄取、解析与来源元数据

## Loader 只负责文档 IO

加载层接收文件路径，输出统一 Document 或明确失败，不进行 embedding、排序和联网策略。职责收窄后，Markdown、TXT 与 PDF 的解析可以单测，索引层只依赖 text 和 metadata，不需要知道每种文件格式的库调用细节。

## 递归遍历需要稳定顺序

load_directory 对目录内文件排序后依次加载，使相同语料在不同运行中产生一致的文档顺序。稳定顺序不是 chunk_id 的唯一来源，但能让建库统计和调试输出更易比较。仅处理白名单扩展名，未知文件应忽略而不是猜测格式。

## 文本使用 utf-8-sig

Markdown 和 TXT 以 utf-8-sig 读取，可以兼容普通 UTF-8 与带 BOM 的 UTF-8。读取后 strip 去除文件两端空白，但正文内部换行保留给 splitter。遇到非法编码应产生 DocumentLoadError，不能用 errors=ignore 静默丢字符。

## PDF 按页生成 Document

pypdf 从每个有可提取文本的页面生成一个 Document，并在 metadata 中保存从一开始计数的 page。按页切分保留了引用定位，也避免整本 PDF 形成超长字符串。扫描图片 PDF 没有 OCR 时可能提取为空，应明确报告而不是伪造文本。

## source_path 使用相对路径

批量加载时，source_path 取相对于知识库根目录的 POSIX 形式，例如 retrieval/rrf.md。这样 chunk 元数据不会携带开发机绝对路径，跨 Windows 和 Linux 仍可比较。直接调用 load_file 时也可显式传入规范化来源。

## 单文件失败不终止整批

目录加载把失败记录为 LoadFailure(path, error)，继续处理其他文件。建库报告同时返回成功文档与失败清单。容错不等于忽略问题：CLI 应展示被跳过的路径，正式评测运行前则应直接要求失败清单为空。

## 空文档属于显式错误

只含空白的文本文件和无可提取页面的 PDF 都不能生成有效 chunk。若把空文本放入索引，会出现零向量、无意义 BM25 长度和无法解释的来源。Loader 在进入 splitter 前拒绝空文档，使下游假设更简单。

## Metadata 是溯源契约

file_type、source_path 和 page 由 loader 提供，splitter 再补充 doc_id 与 chunk_index。向量库和 BM25 必须保留这些字段，工具才能返回来源。若某层只保存正文，Writer 即使找到正确内容也无法给出可靠引用。

## 解析不等于语义清洗

Loader 可以去除 BOM 和文件两端空白，但不应擅自删除标题、代码块或表格。复杂清洗会改变 gold chunk，并可能丢掉精确术语。若未来需要格式感知转换，应提升语料版本并重新检查 gold。

## 评测前直接检查加载结果

当前单人项目不额外维护文件摘要。正式评测前直接记录文档数、chunk 数、切分参数和语料版本，并确认加载失败清单为空、gold ID 全部存在。这个检查更容易理解，也足以防止当前规模下的误跑。

## 安全边界与资源限制

来自用户的超大文件、压缩炸弹式 PDF 或深层目录都可能消耗资源。演示知识库由仓库维护，风险较低；开放上传时需要文件大小、页数、类型和解析超时限制。扩展名只是初步筛选，不是安全验证。

## Loader 测试关注降级报告

一个典型测试目录同时放入正常 Markdown、损坏文本和不支持的 CSV。断言正常文档仍被加载，损坏文件出现在 failures，CSV 被忽略，并检查 source_path 使用相对形式。该测试不需要网络或真实 embedding。
