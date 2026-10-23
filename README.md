# deepseek-v4.1-flash-1017-a100-deepinfer

> **FINAL：R3 + HF5，2026-10-24 冻结。** 沿用已通过验收的部署与性能配置，CPU KV 二级缓存关闭；本次未加入后续讨论的缓存或调度实验。版本信息见 [FINAL-RELEASE.json](FINAL-RELEASE.json)。

A100/SM80 deployment sources with the official DeepSeek recipe protocol, reproducible acceptance scripts, and reported 8-GPU benchmark results.

面向 **8 × NVIDIA A100 80 GiB（SM80）** 的 DeepSeek V4.1 Flash 1017 部署与适配项目。推理引擎基于固定版本的 **vLLM/SM80**，选择性采用 Deepinfer-SM80 的优化；HTTP 协议、工具转换和流式解析使用 DS 官方 **deepseek-recipe 0.2.0**。

这是独立社区适配项目，未整体替换为官方原生 DeepInfer 引擎。仓库提供运行层源码、覆盖层、固定构建依赖和验收方法；模型权重及大型运行镜像由使用者自行准备。

## 已验收配置

| 项目 | 配置 |
|---|---|
| 模型 | DeepSeek V4.1 Flash 1017；API 名称 `DeepSeek-Flash` |
| 硬件 | 8 × A100 80 GiB / SM80；验收主机约 2 TiB 主存 |
| 并行 | TP8 / EP8 / DP1 / PP1 |
| 最大并发 | C32 |
| 最大上下文 | 524,288 token |
| GPU 显存预算 | `gpu_memory_utilization=0.95` |
| KV | `fp8_ds_mla`，GPU 前缀缓存开启 |
| CPU KV 二级缓存 | **默认关闭：`--cpu-kv-gib 0`** |
| Engram 参数主存卸载 | 保留；与 CPU KV 二级缓存不同 |
| 投机解码 | DSpark 原生 k=5 |
| 近似 YOCO FFN 裁剪 | **关闭** |
| API | `http://127.0.0.1:8005/v1` |

CPU KV 的实验代码保留供研究，未包含在本次成功签收配置中。没有验证更低主存容量、其他卡数或其他 GPU 架构。

## 真机性能记录

**2026-10-23 00:31:31（北京时间）**，部署者回传 R3 + HF4 + HF5 的完整验收、性能诊断与启动验收均为 `PASS`，服务已输出 `Ready`，本地 `/v1/models` 返回 `DeepSeek-Flash`。

### 综合性能

`off` 表示关闭思考，`high` 表示高思考强度。每种模式预热 2 次，再测 10 次单请求和 128 次 C32 请求；每请求固定生成 **1,024 token**，计数包含 reasoning token。

| 模式 | 单请求解码中位数（token/s） | 单请求首 token 中位数（秒） | C32 聚合吞吐（token/s） | C32 请求耗时 P95（秒） |
|---|---:|---:|---:|---:|
| off | **144.99** | 0.26 | **1489.48** | 29.34 |
| high | **109.14** | 0.25 | **1277.68** | 31.18 |

C32 聚合吞吐是全部负载请求的生成 token 数除以整个负载阶段墙钟时间，包含排队、预填充和 trace 证据提取开销；它不是每条请求的速度。

### 文本与代码诊断

每组预热 2 次、正式采样 10 次，每请求生成 1,024 token。

| 工作负载 / 模式 | 解码中位数（token/s） | 首 token 中位数（秒） | 计数核验 |
|---|---:|---:|---|
| text / off | 105.77 | 0.26 | CLEAN |
| text / high | 104.15 | 0.24 | CLEAN |
| code / off | 156.97 | 0.25 | CLEAN |
| code / high | 119.55 | 0.24 | CLEAN |

### 数据来源与范围

- 数值来自部署者回传的真实模型验收摘要；[结构化结果](validation/target-hf5-acceptance-20261023.json)和[回传摘要](validation/target-hf5-acceptance-20261023.user-summary.txt)已去除机器标识与安装路径。完整目标机原始报告和逐请求 trace 尚未公开取回；这些不是公开 CI 重新测得的 GPU 性能。
- 综合压测与文本/代码诊断使用不同提示词，数值应分别比较。计时使用客户端接收时间结合原生引擎 token 证据，不是 GPU 内核计时，也不构成相对旧版本的严格 A/B 加速证明。
- **C32 并发与近 512K 单请求检索分别验收**，不能据此声称 32 个满 512K KV 窗口同时驻留。摘要没有给出长请求的精确输入 token 数。
- 通过的协议范围包括 Responses、Chat Completions、Anthropic Messages、JSON/SSE、工具往返及图文；工具约束沿用官方 prompt/parser，没有另加本地 DSML 逐 token 硬语法。

## 使用前准备

本仓库发布的是运行层源码和构建材料。当前 `Dockerfile` 依赖本地固定 R2 基础镜像：

```text
deepseek-v4.1-flash-1017-a100:20261018-r2
sha256:a63405a72ce202bb8347f17fc2432be0df484b794fba0224fd210b6daff004f6
```

该基础镜像没有在此提供公共拉取地址，仓库也不包含约 8 GB 的完整离线运行镜像。**新机器需要先取得匹配的镜像和 1017 权重；仅克隆仓库不足以从零启动。** 基础镜像、模型元数据及来源见 [DEPENDENCIES.md](docs/DEPENDENCIES.md)。

### 已有受支持运行镜像

本地具备 `manifests/runtime.env` 对应的已验收 R3 运行镜像时：

```bash
git clone https://github.com/keplerzip/deepseek-v4.1-flash-1017-a100-deepinfer.git
cd deepseek-v4.1-flash-1017-a100-deepinfer

# 验证仓库文件、镜像身份并执行离线 CPU 检查。
bash install.sh

# 将路径换成已有的 1017 官方权重目录。
bash start.sh --model-dir /srv/models/DeepSeek-V4.1-Flash-1017 \
  --cpu-kv-gib 0 --port 8005
```

启动先做 GPU 内核与真实模型检查，再执行工具/图文、C32、独立近 512K 检索、完整压测和性能诊断。全部通过后自动发布 API 并输出 `Ready`，全过程可能需要较长时间。部署脚本通过归属标签和操作锁隔离实例；同名旧实例由其原目录的 `stop.sh` 停止。

API 默认只对本机回环与检测到的 Docker bridge 网关/来源开放，未设置 API key。用于公网服务时需自行配置鉴权代理；本项目的默认监听策略面向本机和容器访问。

### 构建运行层镜像

已具备上文精确 R2 基础镜像时，可使用仓库内固定的 recipe wheel 离线构建：

```bash
bash build-runtime.sh
python3 scripts/package_manifest.py .
bash install.sh
bash start.sh --model-dir /srv/models/DeepSeek-V4.1-Flash-1017 --cpu-kv-gib 0
```

`build-runtime.sh` 会更新本地 `manifests/runtime.env`。构建后先重新生成本地文件校验清单，再安装：`python3 scripts/package_manifest.py .`。这条路径复用既有 R2 基础镜像；本仓没有提供它的完整从零构建链。

## 日志与验收结果

```bash
# 持续查看引擎日志；Ctrl+C 只退出查看。
docker logs --tail 100 --timestamps -f dsv411017-deepinfer-engine

# 查看服务状态。
bash status.sh

# 查看部署机生成的完整结果。
python3 -m json.tool runtime/full-acceptance.json
python3 -m json.tool runtime/speed-diagnosis.json
python3 -m json.tool runtime/release-profile.json

# 独立测试会暂停本实例前端并持有操作锁。
bash tests/run.sh full --warmups 2 --single-samples 10 --load-requests 128 --output-tokens 1024
```

`runtime/` 是本地运行状态目录，不进入 Git。独立测试报告在 `runtime/acceptance-*/report.json`；启动报告在上面列出的固定路径。测试口径见 [docs/test-methods.md](docs/test-methods.md)。

## 源码和修复

- `adapter/`：官方 recipe 协议绑定、验收 token trace、可选 CPU KV 适配。
- `source/`：基于固定 R2 源码的 SM80 与实验计算覆盖层；近似 FFN 默认关闭。
- `scripts/`、`tests/`：启动检查、CPU/GPU 数值门禁、真实模型验收与性能测量。
- `vendor/`：固定来源、许可证、recipe wheel、checkpoint tokenizer 和参考源码。
- `configs/runtime.json`：本次基线配置；`manifests/runtime.env`：已验收镜像身份。

已合并 HF1–HF5 的 GPU Graph 设备修正、可选 CPU KV 组映射、API 预建中间件栈兼容和 benchmark tokenizer API 修正。CPU KV 批量注册方案保留为实验实现，成功基线使用关闭状态。详见 [修复记录](docs/HOTFIXES.md)。

发布前在固定运行镜像中复跑 **160 项 CPU 检查**（原 HF5 的 158 项加 2 项公开安装入口回归），全部通过；记录见 [public-source-validation.json](validation/public-source-validation.json)。

公开 CI 只检查源码语法、文件完整性与发布边界；完整 CPU 测试需要受支持运行镜像，真实 GPU/质量/性能验收需要目标硬件。公开 CI 不代表 A100 性能复测。

## 来源与许可

本地部署代码采用 [Apache-2.0](LICENSE)，第三方组件保留各自许可，见 [NOTICE](NOTICE)。

- [vLLM](https://github.com/vllm-project/vllm)、[wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport)：固定推理基础与 SM80 适配，Apache-2.0。
- [HalfVulpes/Deepinfer-SM80](https://github.com/HalfVulpes/Deepinfer-SM80)：选择性参考优化，固定 `2a828fc2122b6fc59805c0c98b6aaf78e7da9b48`，Apache-2.0。
- [DeepSeek recipe](https://github.com/deepseek-ai/deepseek-recipe)：官方协议组件，固定 `935817c66f5d6e64838f419692545f06787e485f`，MIT；上游 `server-py/main.py` 保持原字节。
- DeepSeek 模型/tokenizer：MIT；模型权重自行准备，tokenizer 许可保留在 `vendor/tokenizer/LICENSE`。
