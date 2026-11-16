# R9 验证范围

## 目标机最新性能回传

2026-11-16：benchmark PASS、diagnosis PASS、污染提示 []。详见 performance.md 和 validation/target-r9-performance-20261116.json。来源为部署者最终摘要，未独立取回原始 JSON、trace 或 effective-config。完整 run-status、diagnosis 分组数值、模型质量、工具成功率、实际 512K 与 C32×512K 驻留结果未在本次摘要中提供。

## 构建阶段的三轮检查

固定镜像每轮完成 78 项协议/生命周期 CPU 回归、109 项调度测试、安装源码/原生二进制及实际 CLI 的核对。三轮均通过，记录在 validation/ROUND-1.json 到 ROUND-3.json。调度器和请求对象是真实实现，张量在 CPU；仅绕过 GPU wheel 在 CPU 平台上的配置验证，不替换调度逻辑。

精确采样内核针对 SM80、129280/129281 词表编译出 PTX/CUBIN。交叉编译没有执行 GPU，不能当作数值、质量、无死锁或性能证明。GPU sampler flags 测试因无 GPU 跳过，未计为通过。

离线包封存时检查了镜像 blobs、压缩包文件清单、大小、哈希和解压安装。当前公开源码包含新增的性能记录及文档，因此公开源码校验清单与交付离线包清单分开生成；本次提交没有重编译或替换目标机镜像。

## 公开 CI

scripts/check_public_source.py 检查 JSON/Python/shell 语法、源码清单、固定覆盖层、最新回传数据和公开边界。CI 不启动模型、不执行 GPU 性能复测。完整质量/工具/长上下文测试通过 acceptance.sh 手动执行，启动不会自动运行它们。
