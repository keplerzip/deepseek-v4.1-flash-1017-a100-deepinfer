# R9 测试口径

启动与测试完全分开。看到 Ready 后，手动运行 performance.sh；无需停止 API，不执行 GPU 算子检查或独立 CUDA 上下文初始化。

## 本次回传使用的默认口径

- 模式：off 与 high。
- temperature=0、top_p=1、seed=42。
- 每模式预热 2 次，单发正式样本 10 次。
- C32：并发 32，合计 128 请求，每请求固定生成 1024 token，包含 reasoning。
- 综合 benchmark 后，独立执行 text/code × off/high 诊断，每组预热 2 次、采样 10 次。
- 通过本机 UDS 调用同一官方协议路由，结合受测试标记保护的原生 token trace 核算。外部 8005 API 始终开放。

```bash
nohup bash performance.sh > runtime/performance-r9.log 2>&1 < /dev/null &
tail -n 60 -f runtime/performance-r9.log
```

报告目录 runtime/perf-only-UTC时间-PID/：benchmark.json 包含 benchmarks.off/high 的单发解码、TTFT、C32 聚合吞吐和 P95；diagnosis.json 包含分组样本；run-status.json 是整个脚本的退出状态。JSONL 保留逐请求结果。单发与 C32 按顺序运行，不能并行启动两轮测试。

## 单独的采样负载

sampling-performance.sh 使用 temperature=0.6、top_p=0.95，目录带 -sampling 后缀，用于测精确 top-p 路径。本次公开数据来自默认 temperature=0 流程，未回传采样负载数据。

## 解读状态

PASS、CONTAMINATED、FAIL 按报告原样保留。API 开放时，业务流量会影响吞吐、调度和全局计数；本次 R9 回传为 benchmark PASS、diagnosis PASS、污染提示 []。性能测试不等于工具或质量验收。

## 质量、工具与 512K

acceptance.sh 手动运行官方协议 smoke、工具探针、质量样本和长上下文；可以向 --baseline 提供相同测试定义的基准报告。范围和局限见 validation.md。当前公开性能摘要没有提供这些结果，不因配置上限 524288 就写成实际 512K 已通过。
