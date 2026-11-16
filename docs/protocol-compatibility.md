# R9 官方协议与连接

官方 deepseek-recipe 0.2.0 负责请求转换、消息编码和输出解析，固定源码提交 935817c66f5d6e64838f419692545f06787e485f。vendor/official-recipe/source/server-py/main.py 保留上游原字节。

提供 /v1/responses、/v1/chat/completions、/v1/messages、/v1/models，以及现有工具和多模态请求转换。API 模型名 DeepSeek-Flash，默认入口 http://127.0.0.1:8005/v1；沿用 Docker 网桥来源策略。当前默认 API key 关闭，TCP peer ACL 保留，不改为全网卡监听。

工具定义与参数沿用官方 prompt/parser；没有本地 DSML 逐 token 硬语法，也没有服务端执行工具。R7 完整性边界继续处理不完整工具输出、finish 状态与原生 token usage。错误不会通过删除尾部字符伪装成正确调用。

R9 只在协议入口增加可选的请求优先级 header，转发到 EngineClient；不重写正文、工具参数或官方 SSE 数据。默认优先级为 0，客户端保持既有连接方式即可。

性能脚本临时开启 trace 证据接口用于计数，不关闭或隔离外部 API。本次 benchmark/diagnosis PASS 不是独立的工具成功率报告；本次回传未提供协议、视觉和实际 512K 的完整验收结果。
