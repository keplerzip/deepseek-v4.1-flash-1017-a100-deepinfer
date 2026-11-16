# R9 计算实现

推理运行时是固定 vLLM，主 SM80 实现来自 yaoyaoding 的 #66497。固定树 d423120cdc36d4cf8456f43ddfcc654b0c69c5ea，预编译 wheel 基线 ace415c06892a6119c88c8c57ed6fd761b722e32。它不是整体替换为官方原生 DeepInfer。

| 覆盖层 | 固定提交 | 内容 |
|---|---|---|
| #66792 | 1314b9e637571e2d21a68ecceab38193a6e1876a | 精确 Triton top-k/top-p、调用方与工作区预分配 |
| #67712 | 2806929ad924d6db3ebead0352e09e088f04cc01 | prefill 元数据复用；本地适配实际 Ampere override |
| #65282/#65283 | fe4720517d45cfc47acd81716baabf2cf1f124e1 | 显式请求优先级、抢占和 prefill 避让 |
| HF1/HF2 | 随 source 固定 | DSpark local argmax 关闭；SM80 metadata 格式分派 |

source/r9-overlay/manifest.json 记录组合补丁前后哈希及本地移植说明。source/r8-pr66497、source/r8-hf2 与 source/r7-baseline 是当前实现来源/测试依赖，不是单独的旧部署入口。原生二进制沿用固定镜像的 12 个文件，未重新替换 MoE 数值路径。

TP8 / PP1 / DP1；EP 与 EPLB 关闭。保持 1017、NVFP4 MLA KV、MXFP4 indexer、512K 上限、C32、显存 0.94 与 DSpark k5。近似 FFN、decoder replay、SWA bounded replay、CPU KV 卸载均关闭。

优先级可通过 X-DeepInfer-Priority 提供 -100..100，低值更优先；默认全部为 0，同级请求不互相触发该优先级抢占。最多每步抢占 2 个低优先级请求，首 16 token 阶段可避让低优先级 prefill，不按输入长度自动修改优先级。

采样微基准的算子加速不能视为模型整体加速。本次公开性能口径 temperature=0，没有隔离各项优化贡献。SM80 编译记录与性能回传分开保存；所有源码、排除项和许可证见 SOURCE-PROVENANCE.json、NOTICE 与 ACKNOWLEDGMENTS.md。
