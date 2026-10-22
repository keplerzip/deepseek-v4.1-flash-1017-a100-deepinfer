# SM80 候选门禁

候选仅含 `topk_swa` 和 `mhc_aligned` 两项。完整六文件上游快照放在 `vendor/sm80/upstream/`，仅供审阅；禁止把它当作安装覆盖目录。源片段的回退值均为 `0`；正式镜像显式设置为 `1`，启动脚本先执行真实 GPU 门禁，只有全部验收通过才发布 API。当前正式版还包含 MLA 等额外优化，以下脚本只覆盖这两项。

`scripts/check_sm80_candidate.py` 使用标准库完成静态检查，在最终候选镜像内使用既有 PyTorch/Triton/TileLang 完成真实 GPU 检查，不下载任何依赖。

## 三个独立结果

- `--mode static`：校验 vendor 文件、fragment 和已固定来源的 SHA256、Python 语法及安全默认值。输出 `status=PASS, gpu_status=NOT_RUN` 也不能启用候选。
- `--mode gpu`：要求安装文件 SHA 与 manifest 完全匹配、完整不可变镜像 ID，以及默认 8 张 SM80 GPU。对每卡执行真实算子数值与 shape 检查，全部成功才输出 `status=PASS, mode=gpu`。
- `--mode verify`：只读验证先前 GPU 报告是否匹配当前 manifest、两个安装文件、镜像 ID、GPU UUID/型号/CC/SM 数和候选集合。失败非零，且不会覆盖原 GPU 报告。

退出码：0 为该模式完成；1 为检查失败；2 为缺少 PyTorch 或真实 CUDA GPU而未执行。不存在“CPU测试通过所以GPU默认通过”的路径。

在最终镜像内运行，`--package-root` 必须是包含完整 `vllm/` 的 site-packages 父目录，`--image-id` 由宿主 `docker image inspect --format '{{.Id}}'` 获取后传入。例如：

```sh
python3 scripts/check_sm80_candidate.py --mode static \
  --report validation/sm80-static.json

python3 scripts/check_sm80_candidate.py --mode gpu \
  --package-root /usr/local/lib/python3.12/dist-packages \
  --image-id "$DEEPINFER_RUNTIME_IMAGE_ID" \
  --report /state/sm80-gpu-gate.json

python3 scripts/check_sm80_candidate.py --mode verify \
  --package-root /usr/local/lib/python3.12/dist-packages \
  --image-id "$DEEPINFER_RUNTIME_IMAGE_ID" \
  --report /state/sm80-gpu-gate.json
```

启动器必须先运行 verify 并检查退出码，成功后才能设置 `VLLM_DSV41_DEEPINFER_FUSED_COMBINE=1` 和 `VLLM_DSV41_DEEPINFER_MHC_ALIGNED=1`。不能仅检查 JSON 的 status 字符串，也不能把 static PASS 当作 GPU PASS。

## 真实检查内容

TopK/SWA：采用独立 scalar CPU oracle，与 flag 开启的 fused kernel 逐元素精确比较，并与 flag 关闭的固定 R2 Torch reference 对照。覆盖 1–8192 query rows、多请求/零长度请求、非零 query_start_loc 基址、压缩率 1/2/4、top-k 输入短于请求上限、无压缩候选列、非法 index、128 SWA 窗口、长历史与空输入 shape。

mHC：覆盖 token 数 1/6/12/16/17/32、hidden=5120、hc=4，比较 flag 关闭与开启后全部六个返回值。验证 `capture_aux` ABI、pre_mix/RMSNorm 的有无、有限数值、实际 `(6,8,128)` 选型及 17/32 token 原路径不变。residual/aux 和未变路径要求精确一致；其余 BF16 输出限 rtol=0.005/atol=0.002，FP32 输出限 rtol=0.0002/atol=0.00002。6/12/16 token 的 norm+pre_mix+aux 配置还须通过 CUDA Graph 两次回放且与 eager 精确一致。任何 assert 失败都产生 FAIL。

这份 GPU PASS 只证明所列算子和形状的兼容性。它不代表整模型质量、实际速度提升、C32/512K 驻留、原生图文、DSpark 接受率或与官方 API 的工具行为已验收；这些需要单独端到端报告。
