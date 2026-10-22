# 2026-10-22 性能筛选与合入记录

目标：8×A100 SM80、1017 原始权重、TP8/EP8、C32、524288、DSpark k5、GPU 预算 0.92、官方 recipe 工具协议。此次检查官方 vLLM 的 V4.1/SM80 跟踪、性能 PR、vllm-backport、Deepinfer-SM80 及原生 KV offload 文档，并与固定镜像源代码对照。不是互联网穷尽检索。

## 本轮新增：ARC 主存缓存策略

vLLM 原生 CPU KV 层支持 LRU 和 ARC，无需增加依赖或修改数据搬运内核。[官方配置说明](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/#kv_connector_extra_config-reference)

本包改为默认 ARC，并提供 `--cpu-kv-policy lru` 对照。ARC 将近期使用与重复使用的块分开维护，目标是让一次性长请求不容易挤掉经常复用的多轮前缀。数据仍按原格式传输，没有裁剪、再量化或改变采样。

使用镜像中真实策略类运行 16 槽 CPU 对照：热门对话夹杂扫描访问轨迹共 1952 次访问，LRU 命中 508 次（26.0%），ARC 命中 668 次（34.2%）。稳定工作集 800 次访问，两者均命中 792 次（99%）。两种策略的传输中块保护、protected 集合、失败批量淘汰原子性和 clear 检查通过。脚本：`scripts/check_cache_policy.py`；报告：`validation/cache-policy-comparison.json`。

**这些是策略元数据的合成轨迹，不是 A100 吞吐、TTFT 或真实业务命中率测量。** 策略收益受工作负载影响，LRU 可用于回退和同机对照；正式启动仍要求真实 CPU 回载与输出一致性验收。

## 已具备或保留的优化

| 项目 | 固定镜像/本包核对结果 | 决定 |
| --- | --- | --- |
| NUMA 分布 | `shared_offload_region.py` 已有 `_interleave_across_numa_nodes` 和 `VLLM_KV_OFFLOAD_NUMA_INTERLEAVE=auto`；在触页前判断区域大小、在线节点和约束 | 保留原生自动判断；没有目标机拓扑证据时不强制绑定或全节点 interleave |
| token-to-request 元数据复用 | `sparse_swa.py` 已持有复用 buffer，并通过 common metadata 生成映射 | 保留，不将上游同类 PR 再算成新收益 |
| TopK/SWA、mHC、UE8M0、统一 MLA | 现有函数级移植与精确源 hash 已审计；各自有 CPU/编译和目标 GPU 检查 | 保留 |
| Engram 旁路预取 | 现有 stream/event 接线保留 | 通过目标模型工具、视觉、并发和长上下文测试验证，CPU 结果不替代 GPU |
| DSpark local argmax、FULL_DECODE_ONLY | 已在锁定基线中启用 | 保留 k5、标准拒绝采样和当前图形状 |

上游 token-to-request 映射优化：[PR #47474](https://github.com/vllm-project/vllm/pull/47474)。稀疏 TopK 元数据优化：[PR #52084](https://github.com/vllm-project/vllm/pull/52084)。这些 PR 的内核或整机收益不迁移为本包 A100 测量值。

## 本轮未默认加入

| 候选 | 判断与理由 |
| --- | --- |
| 更换整套最新 SM80 分支/镜像 | [官方 SM80 跟踪 #57144](https://github.com/vllm-project/vllm/issues/57144) 中包含新 A100 分支报告；另有 [vllm-backport](https://github.com/wtdcode/vllm-backport) 的 V4.1 路径。涉及版本、模型路径、ABI、量化布局与 Graph 差异，不在正式包中无验证整体替换 |
| 调度预算 8192→16384、FULL_AND_PIECEWISE | backport 示例包含更大预算及另一种图模式；本包保留已固定的 8192 扫描/blocked prefill 与 FULL_DECODE_ONLY。扩大会改变显存、kernel 形状和调度，需要单独 A100 A/B |
| DSpark k6/k7、自适应验证 | backport 文档提示 Ampere 的共享内存和 native draft block 限制；本包保留 k5，不能把 SM100 测量套到 A100。[参考](https://github.com/wtdcode/vllm-backport#deepseek-v4-flash-preview-0731-vision-exp) |
| offload chunk 由 1 改 4 | 原生支持，但混合 KV 组的有效 token 粒度不同；更大 chunk 可能提高 DMA 效率，也会降低短前缀命中粒度。没有实际 A/B 前保留 `blocks_per_chunk=1`。[参数说明](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/#kv_connector_extra_config-reference) |
| FP4/INT4 KV 再压缩、IndexCache 近似、YOCO FFN 裁剪 | 会引入精度或模型行为风险；正式版不启用 |
| 强制 NUMA/线程/PCIe 亲和性 | 需要目标机 GPU↔CPU 拓扑与 NUMA 空闲分布，不能仅凭总内存容量决定 |
| 为新功能换 LMCache/Rust 前端 | 当前内置原生 CPU KV 与官方 recipe 已有可用接线；额外前端/依赖/IPC 需要另行完整适配与离线验证 |

社区 [Deepinfer-SM80](https://github.com/HalfVulpes/Deepinfer-SM80) 本轮检索仍为单初始提交；当前固定提交已有选择性采用，未发现可直接追加的新版本补丁。官方 [V4 性能任务清单](https://github.com/vllm-project/vllm/issues/45861) 提供多种方向，但不能把 V4、Hopper 或 Blackwell 的硬件特化实现当成 V4.1 SM80 的即插即用补丁。

CPU KV/DSpark 上游存在多组命中协调的历史问题，[例如 #47890](https://github.com/vllm-project/vllm/issues/47890)。它不能证明当前固定 V4.1 已受影响或已修好，因此本包不盲目跳过草稿组，而以清 GPU 后的 CPU 真回载与输出一致性作为发布 API 前的验收条件。
