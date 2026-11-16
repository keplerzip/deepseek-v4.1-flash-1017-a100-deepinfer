# R9 依赖与离线材料

## 固定运行镜像

- tag：deepseek-v4.1-flash-1017-a100-deepinfer:20261115-r9
- 构建镜像 ID：sha256:69166b5d0f234671d732ddbd76ca2af91e321b8ca826ecedc22beb4acf13e8f9
- manifests/runtime.env 还记录保存镜像的 index/manifest/config 摘要，兼容 Docker 存储后端差异。
- 完整离线包：DeepSeek-Flash-1017-A100-DeepInfer-20261115-R9-runtime.tar.gz，10424037078 字节；SHA256 a641c8ec0aa9775bc7a5bad23d569aada0746227304bdca6da23e63413ee86d5。

该压缩包、运行镜像、模型权重、大型框架 wheel 与预编译内核不上传 Git。仓库未提供公共镜像拉取地址；仅 git clone 不足以离线安装。完整包的 Docker save 包含镜像所有层，安装时无需另装基础镜像。

## 构建路径

Dockerfile / Dockerfile.r9 以固定 R8 镜像 deepseek-v4.1-flash-1017-a100-deepinfer:20261113-r8-pr66497-8005 为基础，只覆盖封存的 Python/Triton 源码。build-runtime.sh 使用 Dockerfile.r9。构建需要本地已有匹配基础镜像；底层框架 wheel 清单在 vendor/r8-requirements.lock，完整来源和二进制摘要见 SOURCE-PROVENANCE.json。

公开仓库保留官方 recipe 0.2.0 wheel、原始 recipe 代码、1017 tokenizer、GSM8K 测试数据及许可；这些材料不是模型权重。设备 JIT 缓存首次启动时可能重新生成。

## 1017 模型

权重从 DeepSeek 官方 Hugging Face / ModelScope 取得，或复用已有目录。模型清单在 vendor/model-manifest.json。2026-11-15 的发布清单比较发现 48 分片 SHA256/大小未变，仅新增 chat_template.jinja；核查记录在 vendor/official-model-updates/CHECK.json，模板仅供参考，不切换当前服务的官方 recipe 编码路径。

CPU KV 卸载关闭；Engram 参数依赖主存。具体显存和主存可用容量受模型、并发与设备初始化影响，本次性能记录来自既定 8×A100 机器，未承诺其他设备配置。
