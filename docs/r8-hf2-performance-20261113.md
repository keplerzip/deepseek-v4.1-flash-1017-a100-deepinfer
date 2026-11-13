# R8 HF2 · 2026-11-13 性能快照

**冻结标签：`r8-hf2-20261113`。整体 `FAIL`，benchmark `FAIL`，diagnosis `CONTAMINATED`。** 按部署者要求发布当前源码和有污染的实测记录，保留全部状态，不将其改为 PASS 或干净的加速结论。

数值来自部署者回传摘要。[结构化记录](../validation/target-r8-hf2-performance-20261113.json) 与[去除机器安装路径后的摘要](../validation/target-r8-hf2-performance-20261113.user-summary.txt) 随仓库保存。完整目标机 JSON、逐请求 trace 与有效配置尚未独立取回。benchmark 完成于北京时间 **2026-11-13 13:46:14**，diagnosis 完成于 **13:52:05**。

## 配置和口径

既定硬件为 8 × A100 80 GiB；随包配置为 1017、TP8 / PP1 / DP1、EP 与 EPLB 关闭、C32、上下文上限 524288、显存预算 0.94、DSpark k5。CPU KV 卸载关闭，Engram 参数主存卸载保留。官方 deepseek-recipe 0.2.0、R7 完整性处理及原 8005 连接方式保留。目标机的 effective-config.json 未包含在回传摘要中。

随包 performance.sh 在 off/high 各预热 2 次，再运行 10 个单发样本及 128 个 C32 请求，每次固定生成 1024 token；text/code 诊断各预热 2 次、采样 10 次。API 保持开放，测试期间的其他请求可能进入全局计数。thinking 的 completion token 包含 reasoning；单发解码率与 C32 聚合吞吐是不同指标。

## 本次观测值

| 模式 | 单发解码中位数（token/s） | 单发 TTFT 中位数（秒） | C32 聚合吞吐（token/s） | C32 请求耗时 P95（秒） |
|---|---:|---:|---:|---:|
| off | 314.54 | 0.20 | 2145.34 | 17.39 |
| high | 188.88 | 0.23 | 1554.46 | 22.87 |

| 诊断项 | 解码中位数（token/s） | TTFT 中位数（秒） | 样本 | 测量状态 |
|---|---:|---:|---:|---|
| text/off | 249.41 | 0.21 | 10 | NOT_SIGNOFF |
| text/high | 231.59 | 0.18 | 10 | NOT_SIGNOFF |
| code/off | 329.74 | 0.19 | 10 | NOT_SIGNOFF |
| code/high | 265.08 | 0.18 | 10 | CLEAN |

code/high 的单组 CLEAN 不会改变整轮 FAIL/CONTAMINATED。不同诊断提示词也不能与综合 benchmark 的单发速率混用。

## 与 R3 HF5 的数值对照

R3 来自仓库已有的 [PASS 回传记录](../validation/target-hf5-acceptance-20261023.json)。以下只是历史摘要之间的数值比例：R8 有统计污染，且不存在同一时段隔离流量的严格 A/B，不能作为已验收加速比。

| 模式 | R3 单发（token/s） | R8 单发观测值（token/s） | 观测比例 | R3 C32（token/s） | R8 C32 观测值（token/s） | 观测比例 |
|---|---:|---:|---:|---:|---:|---:|
| off | 144.99 | 314.54 | 2.17× | 1489.48 | 2145.34 | 1.44× |
| high | 109.14 | 188.88 | 1.73× | 1277.68 | 1554.46 | 1.22× |

## 失败和污染证据

- off 阶段：全局生成计数 **150585**，测试 usage 合计 **143360**，差额 **7225** token。
- 全程：全局生成计数 **332178**，测试 usage 合计 **286720**，差额 **45458** token。
- high 阶段报告服务繁忙或计数未稳定；这一行回传文字有截断，缺失部分不作补写。
- 原提示列出的可能性包括其他流量、请求遗漏、计数器重置或统计错误，实际原因尚未确定。仅凭这份摘要不能确认所有测试请求均成功。

本次性能记录没有给出模型质量、工具调用成功率或实际 512K 长上下文的验收结果；上下文上限 524288 是配置值，不等于本次 benchmark 验证了该长度。也没有证明 C32 × 512K 同时驻留。

## 固定实现与修复

- vLLM PR #66497，固定工作树 `d423120cdc36d4cf8456f43ddfcc654b0c69c5ea`；PR 比较基线/官方 wheel 基线 `ace415c06892a6119c88c8c57ed6fd761b722e32`。固定 HEAD 的直接父提交是 `d33c17cb78ef59a6242891cebc3dddd7b5e2301c`。
- HF1：关闭 DSpark 未实现 get_top_tokens 接口所对应的 local argmax reduction，保留 greedy / standard 验证和 k5，并挂载实际配置。
- HF2：通用 metadata builder 增加按 backend 分派的格式解析入口；仅 SM80 专用 builder 使用配套 MXFP4 路径。通用 Blackwell/CDNA 检查保留，数值内核及官方协议解析器保持原固定版本。
- 本地 HF2 检查 **17 项 CPU 测试 PASS**，覆盖真实 metadata 构造、原错误复现、无效设备/格式拒绝、归属检查及重载流程；使用 CPU 张量与模拟硬件能力，不冒充 GPU 数值测试。

完整贡献来源见 [致谢](../ACKNOWLEDGMENTS.md)。
