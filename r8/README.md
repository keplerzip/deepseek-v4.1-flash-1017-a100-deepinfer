# DeepSeek 1017 · A100 · R8 HF2 / PR #66497

**2026-11-13 固定快照，标签 `r8-hf2-20261113`。** 本次目标机回传整体/benchmark **FAIL**、diagnosis **CONTAMINATED**；性能数据按部署者要求公开，保留原状态。见[性能记录](../docs/r8-hf2-performance-20261113.md)、[结构化数据](../validation/target-r8-hf2-performance-20261113.json)及[致谢](../ACKNOWLEDGMENTS.md)。

独立 R8 目录，基于 R7 的服务协议与部署约束，适配 [vLLM #66497](https://github.com/vllm-project/vllm/pull/66497)。不覆盖、停止或更新现有 R7。复用原 1017 权重软链接，包内不含权重。

**Git 中提供源码、锁定依赖清单和小型 HF2 更新；大型运行镜像、202 个离线依赖 wheel、完整上游源码压缩包和预编译产物不进入 Git。** 仅克隆不足以从零启动。使用原完整离线 R8 包，或自行准备 `manifests/runtime.env` 对应镜像。目标机已有生成请求的回传记录，但缺少独立取回的完整报告；质量、工具成功率和实际 512K 验收尚未提供。初始本地构建记录保留在 `validation/initial-build-validation.json`，它早于 HF1/HF2 和目标机回传，不能代表当前完整验收。

## HF1 / HF2 与来源

- HF1 关闭 `use_local_argmax_reduction`：当前 DSpark 模型没有 `get_top_tokens()`，保留 k5、greedy 和 standard 验证。配置只读挂载到实际引擎入口。
- HF2 为 metadata builder 增加按 backend 分派的 cache-format 解析；SM80 builder 使用配套 MXFP4 路径，通用硬件检查保留。两个修改文件和原始文件见 `source/r8-hf2/`。没有改动数值内核或官方工具解析器。
- 累计补丁：`updates/R8-HF2-Indexer-20261113/apply.sh`；base64 命令：`updates/R8-HF2-one-command.txt`。默认命令路径针对原离线安装位置。向其他目录应用时显式传入安装目录。
- 核心 SM80 计算与通信实现来自 **Yaoyao Ding / yaoyaoding 的 vLLM PR #66497**。本项目负责部署整合、固定版本、协议兼容及这些本地启动修复。

## 与 R7 的差异

| 项目 | R7 | R8 |
|---|---|---|
| 协议 | 官方 deepseek-recipe 0.2.0，R7 工具流完整性修复 | 保留相同 wheel、官方服务源码与 R7 完整性实现 |
| vLLM | 0.13.0，恢复原 R3 Python 计算源码 | 官方预编译 0.34.0rc1.dev253 / `ace415c06`，叠加 PR `d423120c` 的 30 个运行时 Python 文件 |
| 并行 | TP8 / EP8 / PP1 / DP1 | TP8 / EP 关闭 / PP1 / DP1；按 PR 的通信支持条件关闭 EPLB |
| 注意力、Indexer | R3 的 SM80 Triton 路径 | PR 的 CuTe DSL SM80 路径及 MXFP4 Indexer |
| KV | 原 fp8_ds_mla 路径 | 528B MXFP8 SWA、288B NVFP4 压缩 KV、68B MXFP4 Indexer；使用 PR 配套布局 |
| 矩阵与通信 | R3 已验收路径 | PR 的小批 FP8 GEMV、mHC、Marlin padding、TP 通信优化；由实际拓扑决定可用路径 |
| 1017 / 长度 / 并发 | 原权重 / 524288 / C32 | 保留；512K 表示单请求最大上下文，未声称 32 个 512K 请求同时驻留 |
| DSpark | greedy k5、standard 验证 | 保留；不使用 synthetic acceptance，不带额外 #61869 补丁 |
| GPU 显存利用率 | 0.94 | 0.94 |
| CPU KV / 近似 FFN | 关闭 | 关闭；同时显式关闭新版默认的 SWA/decoder 窗口回放、KV sharing fast prefill；Engram 表仍在主存，THP 关闭 |
| 容器 / 端口 | 原容器 / 8005 | `dsv411017-deepinfer-r8-engine`、`dsv411017-deepinfer-r8-api` / 默认 8005 |

该 PR 使用新版框架接口，R8 配套升级 vLLM/Torch/CuTe 依赖，不能声称“仅替换几个算子”。没有将 PR 硬接入旧 vLLM，也没有复用 R5 的定制路由、ShardQ 或缓存补丁。旧版未使用的本地 DSML grammar/HTTP 服务和 CPU KV 卸载入口已从 R8 移除，避免误走另一套工具解析链路。源码和二进制均固定版本，完整版本清单见 `vendor/r8-requirements.lock`。

## 安装与启动

解压完整压缩包，进入 R8 文件夹：

```bash
sudo -v
bash install.sh
ln -s /你的现有1017权重目录 models/DeepSeek-V4.1-Flash-1017
```

两版可同时保存在磁盘；同一套 8 张卡不能同时加载两份完整模型。先在原 R7 目录手动停止 R7，再启动 R8。R8 启动器发现原引擎运行会退出，不会自行停止它。

```bash
nohup bash start.sh > runtime/start-r8.log 2>&1 &
tail -n 80 -f runtime/start-r8.log
```

也可用 `bash start.sh --model-dir /现有1017目录 --port 8005`。启动只做很快的模型元数据身份检查、端口检查和正常模型加载，不运行设备、算子、质量或性能测试。看到 `Ready: http://127.0.0.1:8005/v1` 即可访问。首次 JIT 属于模型初始化，不能用跳过检查的方式省去。

```bash
sudo docker logs --tail 100 --timestamps -f dsv411017-deepinfer-r8-engine
bash status.sh
bash stop.sh
```

`stop.sh` 只处理属于本 R8 安装的两个容器。网络继续仅允许 localhost 与已核验的 Docker 网桥来源，不增加 API key，也不监听所有外部网卡。默认沿用 R7 的 8005、API 路径、Docker 网桥入口与鉴权方式，客户端连接配置无需修改。

## 先保存 R7 质量基线

R7 尚在运行时，只安装 R8 镜像，不启动 R8；在 R8 目录执行：

```bash
bash acceptance.sh --label R7 --base-url http://127.0.0.1:8005/v1 --gsm-limit 200
```

该命令用 CPU HTTP 客户端向原服务发送测试请求，不改 R7 文件、配置或容器。报告写在 **R8 目录**的 `runtime/`，包括 `quality-R7.json`、`tools-R7.json`、`smoke-R7.json`。不运行生成的命令或代码，不读取用户对话。测试可能持续较长时间；不要与性能测试同时运行。

## R8 质量、工具调用与长上下文验收

R8 Ready 后执行：

```bash
bash acceptance.sh --label R8 --base-url http://127.0.0.1:8005/v1 \
  --gsm-limit 200 --baseline /state/quality-R7.json
```

测试通过公开 HTTP API，保持 API 可用，不创建第二个 CUDA 推理进程。包含：

- 官方 Responses、Chat Completions、Messages、多轮工具结果、SSE、思考计数与合成图片 OCR 冒烟。
- Responses / Chat 工具测试 off 强制调用与 high 自动选择共 144 次：单发→C32→单发，核对工具名、精确合成命令、JSON、nonce、callID、SSE 终止，覆盖引号、XML、注释、heredoc；绝不执行模型返回的命令。
- 固定随机种子选取 200 道 GSM8K，off / high 各一遍；64 次代码推演；64 次 C32 请求局部标记隔离。
- 32K、128K、256K、接近 512K 的三个位置检索；最长输入目标 523776 tokens，保留输出预算，核对官方编码与服务端 usage 的真实 prompt token 数。
- R7/R8 同题、同请求哈希、同采样设置配对比较，列出下降和改善的题目。缺失题目、请求不一致、截断、协议错误或未完成 512K，均不能得到配对 PASS。

完整 GSM8K 使用 `--gsm-limit 1319`，R7/R8 必须使用相同数量。200 题是抽样回归，不构成“完全不降智”的证明；即使总分持平，也要检查 `paired_regressions` 中的具体失分题目。质量、工具或 OCR 失败时，保留 R7 作为正式服务，不因性能更高就替换它。

```bash
python3 -m json.tool runtime/acceptance-R8.json
python3 -m json.tool runtime/compare-R8.json
python3 -m json.tool runtime/tools-R8.json
```

默认两个思考模式共 532 条质量请求，加 144 条工具请求与冒烟请求；完整 GSM8K 是 2770 条质量请求。长上下文测试是单发，不代表 C32 × 512K 容量验收。公开 GSM8K 的来源、提交、文件哈希和 MIT 许可证均随包提供。

## 性能测试单独运行

```bash
bash performance.sh
```

沿用之前的 serving benchmark / diagnosis 工作负载，API 保持开放，不运行设备或算子检查。结果在 `runtime/perf-only-*/benchmark.json` 和 `diagnosis.json`。其他流量会污染计数，不能直接把污染样本用于版本性能比较。

## 回退

在 R8 目录执行 `bash stop.sh`，再到原 R7 目录用原启动命令启动。R8 不迁移/覆盖权重、R7 镜像、状态或容器，不需要给 R7 应用反向补丁。

## 源码与验证

- PR 固定提交：`d423120cdc36d4cf8456f43ddfcc654b0c69c5ea`。
- PR 比较/合并基线及官方 wheel 基线：`ace415c06892a6119c88c8c57ed6fd761b722e32`；官方 wheel 原生二进制与该基线对应，PR 没有原生 C++ 修改。
- `source/r8-pr66497/manifest.json`：基线/改后逐文件 SHA256；构建时检查原 wheel 文件后才安装补丁。
- 原离线包中的 `vendor/upstream-pr66497-source.tar.gz` 为固定完整源码；`vendor/r8-wheels/` 为锁定依赖。这两个大型输入不进 Git。构建输入说明见下方。
- 原完整离线包另有 R7 对照材料；本源码快照保留实际 R8 执行入口、PR 原文和 HF2 的前后对照。
- 原完整离线包附 55 个代表变体的 110 份 PTX/CUBIN 交叉编译产物，本 Git 快照不附带它们；交叉编译不证明目标 GPU 数值正确、工具可靠性或 512K 容量。HF2 的 17 项 CPU 检查记录在 `validation/HF2-BUILD-VALIDATION.json` 和 `validation/hf2-validation.log`。


## 从源码准备缺失的大型构建输入

需要 Linux x86_64 / Python 3.12 以及 Dockerfile.r8 指定的私有本地 R5 工具链基础镜像。该镜像没有公共下载地址；它只用于组装清理后的新 R8 环境，并不作为 R8 的模型计算源码。新机器需要自行取得或重建匹配的输入。

```bash
# 在本目录获取固定 PR 工作树；不跟随 PR 最新 HEAD。
curl -fL --retry 3 https://codeload.github.com/vllm-project/vllm/tar.gz/d423120cdc36d4cf8456f43ddfcc654b0c69c5ea \
  -o vendor/upstream-pr66497-source.tar.gz
python3 -m pip download --require-hashes --no-deps \
  -r vendor/r8-requirements.lock -d vendor/r8-wheels
bash build-runtime.sh
python3 scripts/package_manifest.py .
```

构建使用官方 `ace415c06` wheel，先验证/安装固定 PR 覆盖层，再安装明确标注的 HF2 修复。固定 HEAD 的直接父提交为 `d33c17cb78ef59a6242891cebc3dddd7b5e2301c`；`ace415c06` 是 PR 比较基线，二者含义不同。构建后镜像摘要会变化，需要用新生成的 manifest 安装启动；源码构建不等于重现原目标机性能。

发布前在固定镜像上，以只读挂载的公开快照执行 **82 次 CPU 测试**，覆盖官方协议受控输出、metadata 和重载流程。该计数包含复用的生命周期用例，不代表 82 个 GPU 测试。日志及摘要见 [public-source-validation.json](validation/public-source-validation.json)，目标机污染性能记录仍单独保留。
