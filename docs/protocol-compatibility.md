# 正式版官方协议入口

默认入口改为 `adapter/official_server.py`。官方 `server-py/main.py` 原样保存，全部协议 handlers、转换、prompt rendering、stream parsing、SSE framing 和 accumulator 均来自官方固定源码，出处与 SHA256 在 `vendor/official-recipe/provenance.json`。

本地代码只注入 EngineClient inference、1017 tokenizer / image spelling、原生视觉展开和后端资源清理。没有调用旧版 ProtocolStream、工具参数校验/修复或自写 strict DSML grammar。官方 ConversionOptions 的 reasoning 默认行为 FillEmpty 保持原样。

工具 strict / required / named 采用官方 prompt 和 parser；官方公开库没有约束采样器，工具输出不获得强制 schema 保证。JSON/regex 文本约束接入 vLLM；服务端工具执行、搜索数据与云端在线服务能力不包含在本包。

实际默认入口的 CPU ASGI 检查覆盖三协议 JSON/SSE、工具回传、namespace、custom apply_patch、客户端 tool_search、strict 参数 schema、图片 token、取消/后端清理和官方错误响应。测试输出由 FakeTokenEngine 提供，不代表模型推理质量验收。

`adapter/protocol.py`、`backend.py`、`constraints.py` 和 `server.py` 作为旧包参考及回归测试保留，默认启动完全不导入这些模块。旧版测试日志属于参考路径，不能冒充官方默认入口的全部验证。

正式版新增启动验收专用的 GPU-only 前缀重置接口；只在 runtime/cache-acceptance-active 存在时可调用，正式发布 API 前关闭，用于验证 CPU KV 真实回载，不改动官方协议 handlers。
