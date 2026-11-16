# R9 当前集成说明

- 主计算基线固定为 vLLM #66497 的 SM80 实现。
- HF1 保持 DSpark greedy draft / standard verification，关闭未实现 get_top_tokens 接口的 local argmax reduction。
- HF2 保持通用硬件检查，为 SM80 专用 metadata builder 分派配套格式路径；组合 prefill 复用时保留该修复。
- R9 纳入 #66792 精确采样、#67712 prefill 元数据复用、#65282/#65283 显式优先级调度。逐文件来源见 source/r9-overlay/manifest.json。
- 保持 R7 工具完整性、finish 与原生 token usage 边界；官方 recipe 原始 server 未修改。
- 保持显存 0.94、8005、原连接与鉴权；CPU KV、近似 FFN 和 bounded replay 关闭。
- 启动只加载模型并等待健康状态；性能脚本独立运行，API 持续开放，无算子检查。

本次发布更新当前源码、文档及 2026-11-16 最新目标机性能回传，不重新构建已交付镜像。
