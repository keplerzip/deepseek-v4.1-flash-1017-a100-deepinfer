# 致谢与贡献边界

本项目依赖以下公开工作，感谢维护者提供模型、协议、框架、算子与测试方法。

- **[DeepSeek](https://github.com/deepseek-ai)**：DeepSeek-V4.1-Flash-1017 模型、tokenizer 与 [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) 官方协议组件。本项目固定 recipe 0.2.0，保留原始 server 源码、MIT 许可和第三方声明。
- **[Yaoyao Ding / yaoyaoding](https://github.com/yaoyaoding) 与 [vLLM #66497](https://github.com/vllm-project/vllm/pull/66497) 贡献者**：当前 SM80 计算与通信路径，包括注意力/indexer、FP8/FP4 软件编解码、BF16 计算、mHC、CuTe DSL 与 TP 通信。固定 d423120cdc36d4cf8456f43ddfcc654b0c69c5ea；原生 wheel 基线 ace415c06892a6119c88c8c57ed6fd761b722e32。
- **[Tomas Ruiz / tomasruizt](https://github.com/tomasruizt) 与 [#66792](https://github.com/vllm-project/vllm/pull/66792) 贡献者**：精确 top-k/top-p Triton 采样、工作区管理及相关调用方更新，固定 1314b9e637571e2d21a68ecceab38193a6e1876a。
- **[Woosuk Kwon / WoosukKwon](https://github.com/WoosukKwon) 与 [#67712](https://github.com/vllm-project/vllm/pull/67712) 贡献者**：prefill 元数据复用，固定 2806929ad924d6db3ebead0352e09e088f04cc01；本地另适配实际 Ampere override。
- **[StevenWang-CY](https://github.com/StevenWang-CY) 与 [#65282](https://github.com/vllm-project/vllm/pull/65282) / [#65283](https://github.com/vllm-project/vllm/pull/65283) 贡献者**：优先级抢占、调度与 prefill 避让；组合固定 fe4720517d45cfc47acd81716baabf2cf1f124e1。
- **[vLLM 团队与社区](https://github.com/vllm-project/vllm)**：模型执行、KV 管理、调度、DSpark 和服务基础设施；保留上游 SPDX 与版权声明。
- **[HalfVulpes/Deepinfer-SM80](https://github.com/HalfVulpes/Deepinfer-SM80)** 与 **[wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport)**：为本项目早期 SM80 适配、性能分析和验证方法提供历史参考。当前数值路径的直接来源以上述固定 vLLM 提交为准。
- **PyTorch、Triton、NVIDIA CUTLASS/CuTe DSL、TileLang、CUDA、NCCL** 维护者：提供张量、编译和通信基础设施，组件各自许可保留。
- **[OpenAI grade-school-math / GSM8K](https://github.com/openai/grade-school-math)**：质量样本与 MIT 数据许可，固定来源见 vendor/eval/SOURCE.json。
- **目标机部署者**：在 8×A100 上部署并回传本次 benchmark PASS、diagnosis PASS、污染提示 [] 的最新性能结果。

本项目的贡献集中在 1017 部署集成、离线组织、来源追踪、官方工具协议接入与完整性处理、HF1/HF2 兼容性修复、Ampere 元数据适配、显式请求优先级转发及性能记录。上游数值内核不作为本项目独立原创工作；本次整机性能也不分摊给任何单个 PR。项目为独立社区集成，第三方项目未为其背书。
