# R9 部署与切换

默认端口 8005，API 名称 DeepSeek-Flash；原 localhost 与 Docker 网桥连接、来源 ACL 和免 API key 设置保持不变。GPU 预算 0.94、C32、512K 上限、DSpark k5；CPU KV 关闭，Engram 主存卸载保留。

## 完整离线包安装

```bash
sudo -v
BASE=/ai/services/deepseek-v4.1-flash-1017-a100-deepinfer-r9
cd "$BASE"
tar -xzf DeepSeek-Flash-1017-A100-DeepInfer-20261115-R9-runtime.tar.gz
NEW="$BASE/deepseek-v4.1-flash-1017-a100-deepinfer-r9"
bash "$NEW/install.sh"
```

安装导入预编译镜像，不跑设备、算子或性能检查。GitHub 源码目录不含 images/runtime-image.tar，需要另外取得完整离线包；构建条件见 DEPENDENCIES.md。

## 从仍在运行的 R8 复用权重并切换

先完成上面的解压和安装，然后整段执行：

```bash
sudo -v
bash <<'BASH'
set -euo pipefail
NEW=/ai/services/deepseek-v4.1-flash-1017-a100-deepinfer-r9/deepseek-v4.1-flash-1017-a100-deepinfer-r9
if docker info >/dev/null 2>&1; then D=(docker); else D=(sudo -n docker); fi
OLD=$("${D[@]}" inspect --format '{{index .Config.Labels "deepinfer.root"}}' dsv411017-deepinfer-r8-engine)
WEIGHTS=$("${D[@]}" inspect --format '{{range .Mounts}}{{if eq .Destination "/models/DeepSeek-V4.1-Flash-1017"}}{{.Source}}{{end}}{{end}}' dsv411017-deepinfer-r8-engine)
[[ -f "$OLD/stop.sh" && -f "$WEIGHTS/config.json" ]]
mkdir -p "$NEW/models" "$NEW/runtime"
ln -s -- "$WEIGHTS" "$NEW/models/DeepSeek-V4.1-Flash-1017"
bash "$OLD/stop.sh"
nohup bash "$NEW/start.sh" --port 8005 --cpu-kv-gib 0 > "$NEW/runtime/start-r9.log" 2>&1 < /dev/null &
echo "R9 startup PID=$!"
BASH
```

新目录的模型软链接应当尚不存在；已有链接时检查其目标后跳过 ln 命令。R9 使用独立容器 dsv411017-deepinfer-r9-engine 与 dsv411017-deepinfer-r9-api，发现旧引擎仍运行会退出，不会代替旧目录停止它。

## 启动、查看和停止

```bash
tail -n 80 -f runtime/start-r9.log
# 出现 Ready 后，查询模型列表。
curl -fsS --max-time 10 http://127.0.0.1:8005/v1/models
# 引擎日志；Ctrl+C 只退出查看。
sudo docker logs --tail 100 --timestamps -f dsv411017-deepinfer-r9-engine
bash status.sh
bash stop.sh
```

启动只执行 1017 元数据、归属和端口检查，随后正常加载模型并等健康状态。API 在模型就绪后发布，无需等待质量或性能测试。首次 CUDA/TileLang/CuTe JIT 属于初始化。stop.sh 只停止并移除本目录拥有的容器，保留镜像、权重与文件；操作锁繁忙时应等待当前启动或测试退出，不删除锁文件。

## 性能测试

```bash
sudo -v
nohup bash performance.sh > runtime/performance-r9.log 2>&1 < /dev/null &
tail -n 60 -f runtime/performance-r9.log
```

API 保持开放，不跑算子检查、不重载模型。完整参数与结果格式见 test-methods.md。
