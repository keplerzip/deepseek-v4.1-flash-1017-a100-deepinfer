# 致谢与贡献边界

本项目的部署能够运行在 A100 上，依赖以下开源项目及维护者的工作。感谢他们公开模型、协议、推理框架、算子和验证方法。

- **[DeepSeek](https://github.com/deepseek-ai)**：提供 DeepSeek V4.1 Flash 1017 模型与 tokenizer，以及 [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) 官方消息编码、工具调用与流式解析组件。本项目固定使用 recipe **0.2.0**，保留其许可证及第三方声明。项目名称中的 DeepInfer 体现适配工作的历史背景；本项目的推理运行时基于 vLLM。
- **[Yaoyao Ding / @yaoyaoding](https://github.com/yaoyaoding) 与 [vLLM #66497](https://github.com/vllm-project/vllm/pull/66497) 的贡献者**：提供 R8 的核心 SM80 计算适配，包括 CuTe DSL 注意力与 indexer、FP8/FP4 软件编解码、BF16 计算、mHC 和 TP 通信优化。R8 固定采用 [`d423120c`](https://github.com/vllm-project/vllm/commit/d423120cdc36d4cf8456f43ddfcc654b0c69c5ea)，完整 SHA 及逐文件摘要见 [R8 来源记录](r8/SOURCE-PROVENANCE.json) 和 [PR 文件清单](r8/source/r8-pr66497/manifest.json)。上游后续更新不自动纳入本快照。
- **[vLLM 团队与社区](https://github.com/vllm-project/vllm)**：提供模型执行、调度、KV 管理、DSpark 推测解码、服务基础设施及原生二进制。本项目保留上游源码中的版权与 SPDX 声明。
- **[HalfVulpes/Deepinfer-SM80](https://github.com/HalfVulpes/Deepinfer-SM80)**：其 SM80 适配、性能分析与公开测试为早期 R3 的选择性优化以及后续研究提供了参考。历史引用固定在 `2a828fc2122b6fc59805c0c98b6aaf78e7da9b48`，许可证与来源保留在 [vendor/sm80](vendor/sm80/)。R8 的主要计算路径来自上面的 vLLM PR，历史参考与当前执行来源分别记录。
- **[wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) 及早期 SM80 适配贡献者**：为 R3 的框架基线与回退路径提供基础；历史版本的来源、许可证及固定提交继续保留。
- **PyTorch、Triton、NVIDIA CUTLASS/CuTe DSL、TileLang、CUDA、NCCL 等项目的维护者**：提供张量运行时、编译器和 GPU 通信基础设施，各组件保留自己的许可证。
- **[OpenAI GSM8K / grade-school-math](https://github.com/openai/grade-school-math)**：提供随 R8 质量回归工具使用的公开数据集。来源提交、文件摘要和 MIT 许可见 [评测数据记录](r8/vendor/eval/SOURCE.json)。

本项目的工作集中在 **1017 部署集成、离线材料组织、固定版本与许可证追踪、保持既有连接方式、官方工具协议的接入与完整性处理、HF1/HF2 兼容性修复，以及验收和性能记录**。HF2 仅修正 metadata cache-format 分派：让 SM80 专用 builder 使用其配套实现，并保留通用硬件检查；相关数值内核沿用固定上游代码。

感谢目标机部署者在 **8 × A100 80 GiB** 上部署、复现问题并回传测试结果。R8 本次回传的 `FAIL / CONTAMINATED` 状态和各诊断组状态均原样保留，详见 [性能记录](docs/r8-hf2-performance-20261113.md)。公开这些观测值不代表上游项目为本部署背书，也不替代本项目自身的质量和工具调用验证。
