# DeepInfer 引擎与 SM80 社区实现源码核查

核查日期：2026-10-21（Asia/Shanghai）。本记录只检查公开源码、Git 元数据、公开 PR 和已有报告；未安装依赖，未执行外部仓库脚本，未运行 GPU 推理。主项目生产文件未改动。

## 固定来源

| 来源 | 固定版本 | 只读检出位置 |
| --- | --- | --- |
| deepseek-ai/DeepInfer | `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`（2026-10-20） | `/tmp/deepinfer-evaluation-20261021` |
| 该版本固定 FlashMLA | `2e5429fc5653bab6e081f09477126f731882a6a9` | 上述目录 `vendor/FlashMLA` |
| 该版本固定 DeepGEMM | `1fc9c66975ee322d628b3288534ca0e9ce12a64d` | 上述目录 `vendor/DeepGEMM` |
| 该版本固定 DeepEP | `93eb6eb238127e96c6d7a4a625a6dad158348509` | 上述目录 `vendor/DeepEP` |
| 该版本固定 DeepSelect | `bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18` | 上述目录 `vendor/DeepSelect` |
| HalfVulpes/Deepinfer-SM80 | `2a828fc2122b6fc59805c0c98b6aaf78e7da9b48`（2026-10-20，单个初始提交） | `/tmp/deepinfer-sm80-eval-20261021` |

## 可直接从源码确定的结论

### 官方 DeepInfer 不能在 A100 上原样运行

这不是只依据 README 的支持表：

1. DeepInfer 稀疏注意力后端直接调用 `flash_mla_sparse_fwd` 和 `flash_mla_with_kvcache`，没有 SM80 通用 fallback。[attention_backend.py:1-3,64-71,125-141](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/attention_backend.py#L1-L3)
2. 固定的 FlashMLA 只编译 `sm_100a`/`sm_103a`，并在 prefill、decode 和 fused 路径运行时检查 `arch.is_sm100f()`。[setup.py:24-27](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/setup.py#L24-L27)、[sparse_prefill.cpp:102-104](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/csrc/api/sparse_prefill.cpp#L102-L104)、[sparse_decode.cpp:190](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/csrc/api/sparse_decode.cpp#L190)。单加 `TORCH_CUDA_ARCH_LIST=8.0` 无法改变这些显式目标和运行时检查。
3. HF 配置加载器强制原生 FP8、FP4 专家、32×32 权重分组和 UE8M0 scales；不是通过传入 `dtype=bf16` 就能切换所有模型计算。[config.py:253-264](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/hugging_face/config.py#L253-L264)
4. DeepGEMM 的 packed FP4 输入检查要求 `arch_major==10`；FP8 dispatch 只有 major 9/10，而 DeepInfer GPU scales 固定为 packed int32 UE8M0。[layout.hpp:66-79](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/csrc/utils/layout.hpp#L66-L79)、[gemm.hpp:117-132](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/csrc/apis/gemm.hpp#L117-L132)、[platform.py:55-63](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/platform.py#L55-L63)。注意 DeepGEMM 某些普通 BF16 GEMM 可转 cuBLASLt，因此不能扩大成“所有 BF16 操作都硬锁 SM100”；真正阻断的是本模型使用的 FP8/FP4、sparse indexer 和 Mega 路径。
5. GPU MoE 默认在 NVLink 域内采用 Mega MoE，Mega Gate 对 GPU 直接开启；底层 Mega Gate 和 sparse metadata 分别有 major 10 断言。[mlp.py:177-188,228](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/v4_modules/mlp.py#L177-L188)、[mega_gate.hpp:158-159](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/csrc/apis/mega_gate.hpp#L158-L159)、[attention.hpp:313](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/csrc/apis/attention.hpp#L313)。mHC 也使用 DeepGEMM/TileKernels，需逐路径移植。
6. 固定 DeepEP 为 V2，要求 Hopper 及更新 GPU，并使用 NCCL Gin、DeepJIT、`EPBuffer`/`EngramBuffer`；本模型的 Engram 直接依赖新版 `EngramBuffer`。[DeepEP requirements](https://github.com/deepseek-ai/DeepEP/blob/93eb6eb238127e96c6d7a4a625a6dad158348509/README.md#L45-L57)、[engram.py:10-14](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/v4_modules/engram.py#L10-L14)。旧 DeepEP 的 A100 工作不能视作直接可替换的同 API 依赖。
7. DeepSelect 默认跳过 sm80/sm90，仅显式编译 sm100a/sm103a。[setup.py:119-127](https://github.com/deepseek-ai/DeepSelect/blob/bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18/setup.py#L119-L127)。其注释允许较轻的编译目标调整，但不能解决前述其他库的架构依赖。

构建栈也与“直接使用当前 A100 成品镜像”不同：顶层要求 Python 3.12、GCC 13；NVIDIA requirements 固定 torch `2.13.0+cu132`、Flash Attention 4 beta；build.sh 重装 NCCL `2.32.3`；FlashMLA 实际 NVCC 检查为 >=13.1，比顶层 README 泛称 CUDA 13 更具体。[requirements-nvidia.txt](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/requirements-nvidia.txt)、[build.sh:16-25](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/build.sh#L16-L25)、[FlashMLA setup.py:108-109](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/setup.py#L108-L109)。这些是来源事实，不表示已经验证对应驱动/torch组合在目标机可用。

### 并行布局与项目目标不同

官方样例默认 EP8、CP1、DP8；`ParallelConfig` 只有 DP/CP/EP/Engram 配置，`model_parallel_size=context_parallel_size`。[offline_inference.py:13-29](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/examples/offline_inference.py#L13-L29)、[config.py:150-182](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/config.py#L150-L182)。注意 `ColumnParallelLora` 名称不能当作已实现 TP 的证据：其内部使用 `ReplicatedLinear`。[mla_attn.py:26-58](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/v4_modules/mla_attn.py#L26-L58)

官方生产文档解释 attention DP 与 MoE EP、prefill EP2–8、decode EP64–128，当前生产不用 TP/CP；开源版却提供 CP 能力，因此必须区分“生产采用”和“源码支持”。[deployment Q&A](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md#L31-L65)

推断：八卡 EP8/DP8 对整体吞吐可能有价值，但不能把官方 64 卡 Wide-EP 图或 HalfVulpes 六卡 TP2/PP3 图外推成本项目 TP8/DP1/EP8 的单流速度。CP8/DP1 能否改善本项目长上下文需另测；它不等同 TP8，且不会自动切分 attention 投影权重。

### 模型相同、服务边界不同

官方默认模型确为 `deepseek-ai/DeepSeek-V4.1-Flash-1017`，支持本地 HF 目录；loader 严格校验 architecture、model_type、shards、shape、dtype，并按本 rank 加载专家，不能仅依据同模型名宣布任意变种权重可用。[examples/_config.py:31-51](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/examples/_config.py#L31-L51)、[loading.py:28-59](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/hugging_face/loading.py#L28-L59)

公开 `LLM` 固定 `offline_inference=True`，输入 prompt 或 token ids，返回 token 流/文本；文本路径直接 tokenizer.encode，没有在此处理 Chat Completions/Responses tools。[llm.py:70-80,90-142,144-165](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/entrypoints/llm.py#L70-L80)。仓库虽有 HTTP profile_server，那是性能剖析端点，不是公开模型服务 API。

README 明确缺完整在线服务的 KV cache system、VE/P/D disaggregation 集成与 DP load balancer。引擎已经有 GPU KV 分配和部分 cache/disaggregation 接口，不能误写成“没有 KV cache”。[README](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/README.md#L31-L35)

“与官方 API 处理逻辑对齐”的明确组件是另行发布的 `deepseek-recipe`，官方文档把 API/token 转换与 DeepInfer kernel/runtime 分别描述。[deployment Q&A:19,35-37](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md#L19)。工具调用体验应优先检查 recipe、模板、history、增量 parser、请求生命周期以及模型数值路径；换引擎不能独立证明等效。

## HalfVulpes/Deepinfer-SM80：有源码的 vLLM 优化候选，非官方引擎完整移植

### 实际交付物

仓库的六个 Python overlay 和 unified diff 全在 `vllm/` 命名空间，涵盖 mHC、TopK/SWA 索引合并、KV gather、Triton sparse attention、metadata 复用。无 DeepInfer scheduler/worker/C++ runtime、无模型权重加载器全树、无独立包装或 serving entrypoint；安装要求已有 vLLM checkout。[patch](https://github.com/HalfVulpes/Deepinfer-SM80/blob/2a828fc2122b6fc59805c0c98b6aaf78e7da9b48/patches/vllm-sm80-deepinfer.patch)、[README installation](https://github.com/HalfVulpes/Deepinfer-SM80/blob/2a828fc2122b6fc59805c0c98b6aaf78e7da9b48/README.md#L166-L179)

它需要不在仓库里的既有 SM80 适配。例如 NVIDIA 模型入口导入 `vllm.models.deepseek_v4_1.ampere.ampere_sparse.DeepseekV41AmpereMLAAttention`；cache_utils 及 microbench 导入外部 `fp8_sm80`。这些文件没有随六文件 overlay 交付。[model.py:129-155](https://github.com/HalfVulpes/Deepinfer-SM80/blob/2a828fc2122b6fc59805c0c98b6aaf78e7da9b48/vllm_sm80_deepinfer/vllm/models/deepseek_v4_1/nvidia/model.py#L129-L155)、[cache_utils.py:34-43](https://github.com/HalfVulpes/Deepinfer-SM80/blob/2a828fc2122b6fc59805c0c98b6aaf78e7da9b48/vllm_sm80_deepinfer/vllm/models/deepseek_v4_1/common/ops/cache_utils.py#L34-L43)

它的完整模型文件保留 FP8/FP4 既有 vLLM loader 和 CPU Engram 填充逻辑，这不表示本次优化新增了这些能力。仓库没有提供基线 vLLM commit SHA、补丁 diff index、torch/triton/tilelang lockfile 或完整基础镜像身份；不能直接证明六个整文件可以覆盖本项目当前模型代码。

### 源码已有的可研究优化

| 变化 | 源码证据 | 本项目引入前应验证 |
| --- | --- | --- |
| mHC post+pre delayed 合并，small-token FMA split fallback `(6,8,128)`，相应 warmup | [tilelang.py:1016-1290](https://github.com/HalfVulpes/Deepinfer-SM80/blob/2a828fc2122b6fc59805c0c98b6aaf78e7da9b48/vllm_sm80_deepinfer/vllm/model_executor/kernels/mhc/tilelang.py#L1016)、[patch:892-1021](https://github.com/HalfVulpes/Deepinfer-SM80/blob/2a828fc2122b6fc59805c0c98b6aaf78e7da9b48/patches/vllm-sm80-deepinfer.patch#L892) | 本项目基线是否已有融合；Engram 边界、MTP hidden capture、PP/TP 返回布局和数值误差 |
| 单 Triton kernel 合并 global top-k 与 SWA indices | [rocm.py:109-263](https://github.com/HalfVulpes/Deepinfer-SM80/blob/2a828fc2122b6fc59805c0c98b6aaf78e7da9b48/vllm_sm80_deepinfer/vllm/models/deepseek_v4_1/amd/rocm.py#L109) | 本项目压缩率、窗口、短序列/空行/partial top-k、不重叠 store 正确性 |
| FP8 LUT + UE8M0 位移解码并直接写 gather 输出 | [cache_utils.py:235-309](https://github.com/HalfVulpes/Deepinfer-SM80/blob/2a828fc2122b6fc59805c0c98b6aaf78e7da9b48/vllm_sm80_deepinfer/vllm/models/deepseek_v4_1/common/ops/cache_utils.py#L235) | UE8M0 边界、NaN/零编码策略、cache layout 和张量 stride |
| Triton sparse attention 统一 512 维；decode warp 数选择和 direct out | [patch:1022-1340](https://github.com/HalfVulpes/Deepinfer-SM80/blob/2a828fc2122b6fc59805c0c98b6aaf78e7da9b48/patches/vllm-sm80-deepinfer.patch#L1022) | 108-SM SXM4 A100 与其 74-SM GA100 不同；TP8 的 head 数也不同；应按实际形状调优 |
| prefill ragged indices 缓存 | [rocm.py:914-955](https://github.com/HalfVulpes/Deepinfer-SM80/blob/2a828fc2122b6fc59805c0c98b6aaf78e7da9b48/vllm_sm80_deepinfer/vllm/models/deepseek_v4_1/amd/rocm.py#L914) | reindex layer 更新、跨请求 metadata 生命周期、复用 key 有效性 |

该 patch **没有**编辑 Chat Completions/Responses、token 编码、tool parser 或工具回合状态机文件，因此不能把工具调用改进直接归因于该仓库的六文件变更。

### 文档中的蓝图不可当作已实现

- `01_YOCO...md` 明确当前 baseline 仍对全部 prefill tokens 执行全部 40 层；后文是如何移植 checkpoint-aware compaction 的设计。[01 doc:13-20,68-85](https://github.com/HalfVulpes/Deepinfer-SM80/blob/2a828fc2122b6fc59805c0c98b6aaf78e7da9b48/docs/01_YOCO_PREFILL_COMPACTION_SM80.md#L13-L20)
- `02_DSPARK...md` 从第 114 行起名为 Concrete Implementation Blueprint；涉及的 confidence、adaptive_verification、speculator、pp_utils 文件不在 patch 中。[02 doc:114-140](https://github.com/HalfVulpes/Deepinfer-SM80/blob/2a828fc2122b6fc59805c0c98b6aaf78e7da9b48/docs/02_DSPARK_MTP_VARLEN_TOPK_SAMPLING_SM80.md#L114-L140)
- `03_DEEPSELECT...md` 第 126 行是 Expected Gains，包含 DeepSelect SM80、indexer sharding 和 fused transform 的预计收益；六文件中没有交付这些完整移植。[03 doc:126-134](https://github.com/HalfVulpes/Deepinfer-SM80/blob/2a828fc2122b6fc59805c0c98b6aaf78e7da9b48/docs/03_DEEPSELECT_ATTENTION_KV_KERNELS_SM80.md#L126-L134)
- `05_TOPOLOGY...md` 把已测 baseline、未交付的内部文件和 `v25` 预计值放在同一表；约 6.5M KV、255s 1M prefill、90–112 prose TPS 等是组合方案推演，不是可复现的当前六文件验收。[05 doc:27-37,80-100](https://github.com/HalfVulpes/Deepinfer-SM80/blob/2a828fc2122b6fc59805c0c98b6aaf78e7da9b48/docs/05_TOPOLOGY_1M_CONTEXT_ANALYSIS_SM80.md#L27-L37)

文档还有不应采信的等价性推论：01 文档把 mask-vs-compact 等价验证扩大成 bounded-replay 与 full-SWA 完全相同，并用“global KV 已覆盖历史”解释无精度影响。[01 doc:55-64](https://github.com/HalfVulpes/Deepinfer-SM80/blob/2a828fc2122b6fc59805c0c98b6aaf78e7da9b48/docs/01_YOCO_PREFILL_COMPACTION_SM80.md#L55-L64)。官方部署 Q&A 自己把 bounded replay 产生的 SWA cache 描述为 approximate，不能用 global KV 存在来证明多层局部残差/attention 的数学等价。

### 性能与功能证据等级

仓库 summary.json 自述六卡 74-SM GA100、TP2/PP3、C4、DSpark5，prefill +20.5%/+24.2%、C4 decode +4.1%，Python code/long-gen 单流 160.8/188.8 TPS；这些是作者报告，不是本项目 A/B 结果。[summary JSON](https://github.com/HalfVulpes/Deepinfer-SM80/blob/2a828fc2122b6fc59805c0c98b6aaf78e7da9b48/benchmarks/results/sm80_benchmark_summary.json)

公开交付只有汇总 JSON、图表、三个 kernel microbenchmark；未见原始端到端请求/响应、完整命令、锁定基线和环境、每次运行日志、对应 API14/14 与 tool-parser16/16 测试源码。不能验证 model snapshot 是不是完整 1017、协议测试的覆盖范围，或“官方一致”。

microbench 也不构成现成验收：

1. baseline 函数从当前安装的 vLLM 导入。例如 combine 的 `torch_combine_topk_swa_indices` 是直接 alias；覆盖 overlay 后重跑，会调用当前已改实现而非固定旧基线。[microbench:38-40,780-809](https://github.com/HalfVulpes/Deepinfer-SM80/blob/2a828fc2122b6fc59805c0c98b6aaf78e7da9b48/benchmarks/microbench_sm80_kernels.py#L780-L809)
2. 多处精度检查只计算并打印 err / Bit-exact，没有 assert，退出码 0 不能当作精度 PASS。[microbench:232-246](https://github.com/HalfVulpes/Deepinfer-SM80/blob/2a828fc2122b6fc59805c0c98b6aaf78e7da9b48/benchmarks/microbench_sm80_kernels.py#L232-L246)、[part2:322-350](https://github.com/HalfVulpes/Deepinfer-SM80/blob/2a828fc2122b6fc59805c0c98b6aaf78e7da9b48/benchmarks/microbench_sm80_part2.py#L322-L350)
3. 主要目标硬件是 74-SM、无 P2P 六卡 GA100，项目是 108-SM A100-SXM4/NVSwitch 八卡。代码部分调参会读取 SM 数，但仍有依赖 shape 的 warp/split 规则；不能按 GPU 同为 SM80 就继承收益。

## 上游 PR 搜索范围与其他候选

GitHub API 于核查时返回 DeepInfer `pulls?state=all&per_page=100` 共 89 条；按 SM80/A100/Ampere/SM90/Hopper 搜索正文和标题。

- [#48](https://github.com/deepseek-ai/DeepInfer/pull/48) 用 2×A100 做 host registration、shared pool、worker method 测试，正文明确没有模型加载或完整推理；不是 SM80 引擎移植。
- [#92](https://github.com/deepseek-ai/DeepInfer/pull/92)，`DarkraiHL/DeepInfer` 的 `sm90-hopper` 分支，API 观测 head `0f7a87221d96648a298359ebe0b13405ed5e659f`。它报告 4/8×H20 端到端、1M prefill、DSpark、CUDA Graph、GSM8K，同时新增 FlashMLA-SM90、humming、FP8 Gluon GEMM、indexer kernels、non-mega MoE、DeepSelect/DeepEP patches，约 +9140/-72 行。它是完整移植工作量的参考，**仅 SM90，不是 SM80 候选**。本次未复核其全部源码或复现性能。
- [#88](https://github.com/deepseek-ai/DeepInfer/pull/88)、[#91](https://github.com/deepseek-ai/DeepInfer/pull/91)、[#101](https://github.com/deepseek-ai/DeepInfer/pull/101) 是 Hopper 算力识别、MXFP4→FP8 转换与加载子步骤；正文自己区分“能加载”和“能运行”，不应作为端到端 A100 支持证据。

搜索不是对所有 fork 的穷尽证明。HalfVulpes 是已找到、具有实际 kernel 修改及 SM80 作者测量的候选；目前证据支持把它列为本项目现有 vLLM 计算路径的可选优化来源，不支持称其“已移植官方完整 DeepInfer”。

## 工程判断与未验证项

直接换官方 DeepInfer：需要新建 SM80 backend，包括 FP8/FP4 dense/grouped GEMM、sparse attention/indexer、gate/mHC、EP/Engram 通信、若干 TileLang 量化/缓存算子，以及并行和在线服务适配。现有 vLLM 的 SM80 算子可以参考，但 tensor layout、量化规则、scheduler/cudagraph contract 不同，不能通过替换几个 Python import 保证一致。属于新的推理后端移植项目，不能给出“几天即上线”或性能承诺。

采用 HalfVulpes 局部优化：合理下一步是基于本项目固定 vLLM 基线逐 hunk 比较、保留已有功能修复、先选低耦合 kernel，在本项目目标 GPU 上做相同请求/输出预算的 A/B，并补数值、C32、512K、图文、DSpark、tool history/streaming 和压力取消验收。无需先换推理引擎才研究官方协议组件。

本次未验证：目标机编译及驱动兼容、GPU kernel 数值、实际权重全量加载、C32/512K 内存、吞吐/TTFT、端到端工具调用成功率与官方 API 逐字段对齐。所有上游 benchmark 和功能 PASS 均保留作者报告属性。
