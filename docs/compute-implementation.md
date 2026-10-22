# 正式版计算实现与限制

固定基础镜像：`sha256:a63405a72ce202bb8347f17fc2432be0df484b794fba0224fd210b6daff004f6`。
社区源码：HalfVulpes/Deepinfer-SM80 `2a828fc2122b6fc59805c0c98b6aaf78e7da9b48`。
官方 recipe：`935817c66f5d6e64838f419692545f06787e485f`，wheel 0.2.0。

`source/sm80-overlay` 是之前两项函数级补丁。`source/beta-overlay` 新增四个基于真实 R2 文件的完整生成文件、精确基线 SHA256 和可审核 diff；构建只允许覆盖对应 R2 字节，不接受不匹配的社区整文件。

社区统一 512 维 partial kernel 保留全部 R2 调用 ABI，仍走原 split/reduce，保留融合 inverse RoPE；没有为了 direct-out 删除既有功能。社区 BF16 scale helper 在编码 0 时使用 0，本包改为位模式 64，即 2^-127；gather 的 f32 对应位模式 1<<22。全部 256 个编码按 f32 / BF16 参考验证。

Engram 依赖当前主 stream 的 hash 输入，在独立 stream 中 lookup(background=True)，记录事件，在 embed 消费前等待。staged_rows 持久保留。该路径应在目标 A100 验证 CUDA Graph 捕获和回放，CPU 测试无法证明跨 stream 正确性。

YOCO FFN 改写默认关闭：launcher、镜像 ENV 和 Python 回退值均为 0。该分支代码保留作研究参考，正式启动入口不允许启用。原近似方案作用于 L21–39 且非 sequence-parallel 的 eager prefill，decoder graphs、producer L20、MTP 层、短请求和无法识别的 metadata 走完整 FFN。压缩行来自每请求尾部及 checkpoint 尾部；MoE padding 临时对齐并在 finally 恢复。所有注意力和共享 KV 仍在完整坐标运行，FFN 缺省行返回零，属于数值近似。它不是完整 query compaction，也没有给近似 SWA 状态声称无损。

默认配置保留另外五项优化，开启 prefix caching，端口 8005。`configs/runtime.json` 与原稳定项目逐键相同。正式版强制关闭近似 YOCO FFN，保持前缀缓存开启。CPU KV 使用原生 OffloadingConnector；容量及验收策略以 README 为准。`reference` 仍使用官方协议入口。完整 YOCO、adaptive MTP、DeepSelect、ZOS 均未声称已经移植。

CPU/ASGI 和离线编译结果不意味着推理成功。包内 GPU A/B 脚本会对真实卡执行 MLA 数值和 graph 检查；另一个脚本覆盖前两项候选。Engram 的整模型实测通过真实模型启动与长上下文/并发测试另行判断。没有继承社区 74-SM / TP2 / PP3 / C4 的性能数字。
