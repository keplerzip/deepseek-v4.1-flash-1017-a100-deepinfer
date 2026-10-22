# R3 测试方法与旧稳定版的对应

本版直接移植旧项目的 `tests/acceptance.py`、`diagnose_speed.py`，适配 DS 官方 recipe 请求与解析。原文件和移植后 SHA256 在 `validation/test-method-provenance.json`。模型响应只来自实际请求；单元测试中的 fixture 仅验证测试程序本身。

| 方法 | 本版保留的验证内容 |
| --- | --- |
| smoke | 唯一模型身份、off/high SSE、单图与三图 OCR、反序工具结果图片关联、冷热前缀 cached_tokens、C32 自动工具调用、DSpark 原生 k5 计数 |
| benchmark | off/high 分别预热、重复单发、C32 负载；报告原生首 token、客户端可见首 delta、decode TPS、端到端 TPS、C32 总吞吐及 P50/P95 |
| long-context | 使用官方转换和实际引擎 renderer 计数，构造 523776 目标输入、允许向下 1024 token 误差、512 输出预算；检索开头、25%、50%、75%、末尾五处，usage 必须与构造计数一致 |
| full | 串行执行上述全部常规套件 |
| diagnosis | text/code × off/high；保存每请求原始指标、token 批次、DSpark 草稿/提议/接受数及位置计数、延迟与批次间隔；不将计数推导当成 GPU kernel 时间或因果提速证明 |
| full-window-c32 | 原版显式压力入口仍保留，发送 32 条独立前缀的近满窗口请求并采样调度轨迹；缺少逐请求 KV 驻留遥测时即使请求成功仍为 NOT_PROVEN，不包含在默认 full 中 |

## 执行与采样量

在停止旧版本并启动本版后：

```bash
bash tests/run.sh smoke
bash tests/run.sh benchmark
bash tests/run.sh long-context
bash tests/run.sh full
bash tests/diagnose-speed.sh --warmups 2 --samples 10 --output-tokens 1024
```

独立 benchmark 默认沿用旧套件：每档预热 1 次、单发 5 次、C32 负载 64 请求、输出 512 token。可传 `--warmups / --single-samples / --load-requests / --output-tokens` 调整。

`start.sh` 除内核、协议、CPU KV 回载和容量 smoke 外，自动运行 `full`：每档预热 2 次、单发 10 次、C32 阶段 128 请求、输出 1024 token；再运行四组诊断，每组预热 2 次、采样 10 次、输出 1024 token。这与之前一键入口的采样规模一致。前端只在全部通过并满足 500 GiB 可用主存后启动。

独立测试持有本安装操作锁，临时暂停本安装前端，用 UDS 发请求，退出后清理验收标记并恢复前端。不要从其他 docker exec 会话同时生成；全局计数核账能发现常见污染，但不能完全排除无生成 token 的干扰。

## 官方协议下的 token 核账与计时

官方 server-py 源码和协议响应字节格式不改。验收请求通过独立头启用有界内部 trace：记录同一次 EngineClient 的原生 token IDs、批次时间、prompt 数、完成状态和请求体 SHA256。客户端必须将 ID、SHA256、原生 token 数与官方 usage 对齐，再核对全局生成计数。trace 溢出、缺失、未完成或不一致一律不能签收。报告保留原生批次数量/时刻、token IDs 指纹及请求体指纹；原始输出 token IDs 只在核账期间使用，不随报告保存。C32 整阶段墙钟时间还包含内部证据提取开销。

官方 SSE 没有旧适配器附加的 token IDs，因此内部 trace 同时记录每次发送原样 SSE body 的累计字节数与原生 token 数。客户端记录接收/解析后的字节偏移和时刻，再绑定两者；拆包、合包不改变 token 总数，同次接收的多个边界合并成一批。所有 token 必须有接收证据，双方响应字节总数必须一致。decode 和 TTFT 继续使用旧版的客户端批次到达口径，包含 HTTP/解析开销；`first_visible_seconds` 单独记录首个可见文本/推理 delta。官方 SSE 分批可能不同，**仍不据此直接宣称严格跨版本提速比例**。事件数、字符数和本地重新 tokenize 均不代替原生输出 token 数。

固定输出负载仅在验收标记和请求头同时启用时使用 `min_tokens=max_tokens`，在预算前屏蔽 EOS；普通 API 请求采样保持原样。旧版使用 `ignore_eos`，本版借助原生最少输出预算实现相同固定 token 规模，报告注明该差异。工具调用仍经官方转换与解析；没有引入旧适配器的 DSML grammar。

trace、临时精确 tokenize 和清 GPU 缓存接口只在验收标记存在时可用。标记清理后返回 404。KV 验收依然要求清理 GPU 前缀后观测到实际 CPU 回载字节，并核对文本/工具参数一致。

## 证据边界

本包 CPU/ASGI 测试、SM80 编译、离线安装均不能代替实际 A100 测试。C32 是调度并发与短请求负载；单路近满 512K 不证明 32 路满窗口同时驻留。统计质量、长期生产 SLA 和实际 App/New API 网关链路不由这些 smoke 自动证明。
