# 历史 R1 选择记录

下文记录此前两项候选的保守版本。BETA1 扩展范围见 `beta-implementation.md`，默认启动策略以根 README 为准。

# Deepinfer-SM80 与固定 R2 的实际源码对比

对比来源是 HalfVulpes/Deepinfer-SM80 的固定提交
`2a828fc2122b6fc59805c0c98b6aaf78e7da9b48`，以及通过一次性、断网、只读容器从项目固定
R2 镜像 `sha256:a63405a72ce202bb8347f17fc2432be0df484b794fba0224fd210b6daff004f6`
读取的真实安装文件。没有操作原服务容器。完整 SHA256、逐文件 patch 检查和差异规模保存在
`validation/r2-sm80-six-file-comparison.json`。

| 上游六文件 | 对 R2 执行原始 patch 的检查 | 采用决定 |
|---|---|---|
| `mhc/tilelang.py` | 文本可匹配 | 不整体采用；会另注册与 R2 专用 mHC 模块同名的 custom op，且上游版本没有 R2 的辅助状态输出 ABI |
| `mhc/warmup.py` | 文本可匹配 | 不整体采用；R2 已有专用预热注册，实验配置复用它 |
| `amd/rocm.py` | 不匹配 | 只抽取 TopK+SWA prefill 合并 kernel，保留原 Torch 参考和其他全部定义 |
| `common/ops/cache_utils.py` | 不匹配 | 暂不采用；R2 已有宽向量 gather，UE8M0 极值指数和 LUT 数值边界需单独审查 |
| `nvidia/model.py` | 不匹配 | 不采用；包含当前 R2 缺失的 PP/Engram 依赖，并退回显式重构草稿辅助状态 |
| `ops/rocm_aiter_mla_sparse.py` | 不匹配 | 不整体采用；会删除现有 fused inverse RoPE、blocked prefill 和 8192 行 fast scan |

上游 `model.py` 顶层导入当前镜像没有的 `pp_share.PPSharePlan`，并调用当前镜像不存在的
`register_skip_weight_names`、`needs_checkpoint_fill`、`fill_pinned_from_checkpoint`
和 `rows_staged_by_runner`。因此上游 README 的六文件直接覆盖命令不适用于这个项目。
其中 PP 和磁盘 Engram 的新路径也不是当前 TP8/PP1、主存 Engram 卸载目标的必要条件。

实际选择的修改位于 `source/sm80-overlay`：

1. `topk_swa`：把 prefill 的 Torch 多步索引组合换为一项可单独开关的 Triton 合并函数。
   store 区间互不重叠，原函数保留为参考；不改 attention 类、图缓冲、工具协议或权重。
   非 Ampere、非正压缩比和非二次幂窗口继续走原分支。
2. `mhc_aligned`：在 R2 已有 fused mHC 的配置选择中，为适用的小 token 形状添加
   `(6, 8, 128)`。保留 `capture_aux`、六项返回值和对应预热，避免重新加入草稿辅助状态的
   单独 post/mean 计算。

两项默认关闭，不宣称已提速。准备工具按原文件 SHA256 重建独立 overlay，并逐函数检查
其余代码未变。CPU 验证已执行 7 项测试，包括实际 Triton CPU interpreter 与原 R2 Torch
的差分；8 个 kernel 元数据案例、1 个兼容 fallback 和空输入通过。4 个专门化配置也已离线
编译为 SM80 PTX/cubin。对应证据是 `validation/sm80-selected-cpu.json`、
`validation/sm80-selected-source-check.json` 和 `validation/sm80-selected-compile.json`。

上述证据只证明源码、控制流、CPU 解释执行和离线编译，不证明 CUDA 并发写安全、mHC GPU
数值、CUDA Graph 回放、八卡整模型质量或性能。独立 GPU 门禁和同机 A/B 仍需执行。
这些计算优化也不实现工具调用；官方协议一致性来自另外的 recipe 适配层及其端到端验收。
