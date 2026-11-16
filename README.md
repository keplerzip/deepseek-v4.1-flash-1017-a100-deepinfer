# DeepSeek 1017 · 8×A100 · R9

当前分支提供 **R9 源码和最新一次性能记录**。2026-11-16 目标机回传：**benchmark PASS、diagnosis PASS，污染提示 `[]`**。完成时间为北京时间 **2026-11-16 20:51:35**。

本项目面向 8 × NVIDIA A100 80 GiB（SM80），运行 DeepSeek-V4.1-Flash-1017。推理基于固定 vLLM 与 SM80 专用实现，工具协议使用官方 **deepseek-recipe 0.2.0**；它是社区部署集成，不是官方原生 DeepInfer 引擎。

[性能记录](docs/performance.md) · [部署和切换](docs/deployment.md) · [测试口径](docs/test-methods.md) · [计算路径](docs/compute-implementation.md) · [协议说明](docs/protocol-compatibility.md) · [依赖](docs/DEPENDENCIES.md) · [验证范围](docs/validation.md) · [致谢](ACKNOWLEDGMENTS.md)

## 最新 R9 性能

| 模式 | 单发解码中位数（token/s） | 单发 TTFT 中位数（秒） | C32 聚合吞吐（token/s） | C32 请求耗时 P95（秒） |
|---|---:|---:|---:|---:|
| off | **363.73** | **0.20** | **2253.57** | **16.69** |
| high | **271.61** | **0.20** | **1841.59** | **19.71** |

`off` 为关闭思考，`high` 为高思考强度。两种模式均使用 temperature=0、top_p=1、seed=42；各预热 2 次，单发 10 次，C32 共 128 请求，每请求固定生成 1024 token。API 在测试期间保持开放，脚本不执行算子检查。

**本次回传没有统计污染提示，benchmark 与 diagnosis 均为 PASS。** [结构化记录](validation/target-r9-performance-20261116.json) 和[回传摘要](validation/target-r9-performance-20261116.user-summary.txt)保留了数值与状态。数值来自部署者回传，本仓库未独立取回完整目标机报告、逐请求 trace 和 effective-config。diagnosis 只回传了 PASS，未提供 text/code 分组数值，因此不补写。该性能记录不等于质量、工具调用成功率或实际 512K 验收结论。

## 运行配置

| 项目 | R9 配置 |
|---|---|
| 模型 / API 名称 | DeepSeek-V4.1-Flash-1017 / `DeepSeek-Flash` |
| 硬件 / 并行 | 8 × A100 80 GiB；TP8 / PP1 / DP1；EP、EPLB 关闭 |
| 并发 / 上下文上限 | C32 / 524288 token |
| GPU 显存预算 | 0.94 |
| KV / indexer | nvfp4_ds_mla / mxfp4，SM80 专用 metadata 分派 |
| 投机解码 | DSpark k5，greedy draft / standard verification，local argmax reduction 关闭 |
| CPU KV 二级缓存 | 关闭；Engram 参数主存卸载保留 |
| API | `http://127.0.0.1:8005/v1`；原 Docker 网桥入口和来源 ACL，免 API key |
| 协议 | 官方 recipe 转换/编码/解析，保留 R7 工具完整性与 finish/usage 边界 |
| 启动 | 正常模型加载与健康检查；性能测试单独执行 |

## R9 纳入的更新

- **#66792**：精确 top-k/top-p Triton 采样及工作区预分配。
- **#67712**：prefill 元数据复用，适配实际使用的 Ampere override。
- **#65282 / #65283**：显式优先级抢占及首 token 阶段的低优先级 prefill 避让。
- 保留固定 **#66497** 的 SM80 计算路径和 HF1/HF2 启动兼容修复。

固定提交、纳入/排除原因见 [SOURCE-PROVENANCE.json](SOURCE-PROVENANCE.json)。没有近似 FFN 裁剪、decoder replay、SWA bounded replay 或 CPU KV 卸载。优先级默认全为 0；可选 `X-DeepInfer-Priority` 请求头使用 -100..100，数值越小越优先，不按上下文长度自动降级。

## 使用

仓库发布源码、覆盖层和固定来源；完整离线镜像与权重不进入 Git。仅克隆仓库不足以从零启动，需要先取得匹配的离线运行包或构建依赖，见 [依赖说明](docs/DEPENDENCIES.md)。

已有完整离线运行包时：

```bash
bash install.sh
ln -s /your/existing/DeepSeek-V4.1-Flash-1017 models/DeepSeek-V4.1-Flash-1017
# 在旧部署目录执行 stop.sh，释放 GPU 和 8005 后：
nohup bash start.sh --port 8005 --cpu-kv-gib 0 > runtime/start-r9.log 2>&1 < /dev/null &
tail -n 80 -f runtime/start-r9.log
```

看到 `Ready: http://127.0.0.1:8005/v1` 即可访问，客户端连接配置保持原样。首次设备 JIT 属于模型初始化。完整切换、日志和停止命令见 [部署说明](docs/deployment.md)。

```bash
# 单独运行与本次回传相同口径的单发、C32 和 text/code 诊断。
nohup bash performance.sh > runtime/performance-r9.log 2>&1 < /dev/null &
tail -n 60 -f runtime/performance-r9.log
```

结果位于 `runtime/perf-only-*/benchmark.json`、`diagnosis.json` 和 `run-status.json`。API 始终开放；其他业务流量可能影响数字并产生污染提示。此默认口径为 temperature=0，不能将它的整机表现归因于 top-p 算子单项优化。

## 官方模型与许可

2026-11-15 对照官方 Hugging Face / ModelScope 清单，48 个权重分片的 SHA256 和大小未变，仅新增 chat_template.jinja。模板作为参考保存，服务继续使用官方 recipe；[核查记录](vendor/official-model-updates/CHECK.json)随源码保留。

本地集成采用 [Apache-2.0](LICENSE)，第三方原始许可与 SPDX 保留，见 [NOTICE](NOTICE) 和 [致谢与贡献边界](ACKNOWLEDGMENTS.md)。当前分支只展示本次 R9 性能记录。
