# R9 正式稳定版固化与维护规则

固化日期：2026-11-16（Asia/Shanghai）。部署者在回传 benchmark PASS、diagnosis PASS、污染提示 [] 后，明确决定将现有 R9 作为正式稳定基线。

## 固定身份

- 稳定标签：`r9-stable-20261116`。
- 稳定分支：`stable/r9`。
- 运行代码基线提交：`b6b3b1c4f55a241fcf58181e3e56ef6c120a9342`。本次固化提交仅登记发布状态和维护规则，不改变运行逻辑。
- 运行镜像：`deepseek-v4.1-flash-1017-a100-deepinfer:20261115-r9`。
- 构建镜像 ID：`sha256:69166b5d0f234671d732ddbd76ca2af91e321b8ca826ecedc22beb4acf13e8f9`。
- 完整离线包：`DeepSeek-Flash-1017-A100-DeepInfer-20261115-R9-runtime.tar.gz`。
- 大小：10424037078 字节。
- SHA256：`a641c8ec0aa9775bc7a5bad23d569aada0746227304bdca6da23e63413ee86d5`。

原压缩包、镜像及构建时的 VERSION/兼容性标识保持原字节；其中 EXPERIMENTAL/candidate 是原构建标识，正式稳定身份以本固化清单和稳定标签为准。现有部署无需重新安装或重载模型。GitHub Release 发布源码与说明，完整运行包和权重未作为公开附件上传。

## 冻结范围

R9 保持 1017、TP8/PP1/DP1、EP 关闭、C32、524288 上限、显存预算 0.94、DSpark k5、官方 deepseek-recipe 0.2.0。端口 8005、连接方式、ACL/鉴权与模型名保持原样。CPU KV 关闭，Engram 参数主存卸载保留；近似 FFN、decoder replay 和 SWA bounded replay 关闭。启动不自动跑性能或算子测试。

## 后续工作

1. R9 的稳定标签和 stable/r9 分支固定，不移动标签、不重写其提交，也不往稳定分支追加功能更新。
2. 新适配、优化、修复和实验从该基线复制到独立的新版本目录、镜像标签和开发分支，例如 R10。未来 main 可以发布新版本，但稳定引用始终指向本版。
3. 不覆盖现有 R9 的 runtime-package、运行配置、镜像标签所指制品、压缩包或目标机部署；新版本通过独立比较后再由部署者决定切换。
4. 性能数据保持原回传值和状态。正式稳定是部署者的版本接受决定，不将未提供的质量、工具准确率、实际 512K 或 C32×512K 验收补写为 PASS。

维护者与后续自动化应先读取根目录 AGENTS.md。
