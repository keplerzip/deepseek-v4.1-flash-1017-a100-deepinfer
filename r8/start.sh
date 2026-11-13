#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
model_dir="$ROOT/models/DeepSeek-V4.1-Flash-1017"
port=8005
while (($#)); do
  case "$1" in
    --model-dir) model_dir=${2:?Missing model path}; shift 2 ;;
    --port) port=${2:?Missing port}; shift 2 ;;
    --cpu-kv-gib) [[ ${2:?} == 0 ]] || fail 'R8 keeps CPU KV disabled'; shift 2 ;;
    --help) printf 'Usage: bash start.sh [--model-dir PATH] [--port 8005]\nIndependent R8 candidate; no device/operator/performance checks at startup.\n'; exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done
[[ "$port" =~ ^[0-9]+$ ]] && ((port >= 1024 && port <= 65535)) || fail 'Invalid port'
[[ -d "$model_dir" ]] || fail "Model directory/soft link missing: $model_dir"
model_dir=$(realpath -e -- "$model_dir")
[[ "$model_dir" != *:* && "$model_dir" != *,* && "$model_dir" != *$'\n'* ]] || fail 'Unsupported model mount delimiter'
exec 9>"$STATE/operation.lock"
flock -n 9 || fail 'Another R8 runtime operation is active'
image
for name in "$ENGINE" "$FRONTEND"; do
  if docker inspect "$name" >/dev/null 2>&1; then
    owned "$name" || fail "R8 name belongs to another installation: $name"
    fail 'Existing R8 container found; use status.sh or stop.sh in this R8 directory'
  fi
done
# Container metadata only: never stop the stable deployment or probe its GPUs.
if [[ $(docker inspect --format '{{.State.Running}}' dsv411017-deepinfer-engine 2>/dev/null || true) == true ]]; then
  fail 'Stable R7 is running on these eight GPUs. R8 did not stop it. Stop R7 yourself before testing R8.'
fi
python3 "$ROOT/scripts/validate_1017.py" "$model_dir"
printf '%s\n' "$model_dir" > "$STATE/model.path"
mapfile -t network_ids < <(docker network ls --filter driver=bridge --format '{{.ID}}')
if ((${#network_ids[@]})); then
  docker network inspect "${network_ids[@]}" > "$STATE/docker-networks.json"
else printf '[]\n' > "$STATE/docker-networks.json"; fi
docker run --rm --pull never --runtime runc --network host --read-only --tmpfs /tmp:rw,size=64m \
  "${USER_ARGS[@]}" -v "$STATE:/state" --entrypoint /opt/r8-venv/bin/python "$IMAGE_ID" \
  /opt/deepinfer/scripts/policy.py --networks /state/docker-networks.json --output /state/api-listeners.json --port "$port"
docker run --rm --pull never --runtime runc --network host --read-only --tmpfs /tmp:rw,size=64m \
  "${USER_ARGS[@]}" -v "$STATE:/state" --entrypoint /opt/r8-venv/bin/python "$IMAGE_ID" \
  /opt/deepinfer/scripts/check_port.py
[[ ! -e "$STATE/api.sock" || -S "$STATE/api.sock" ]] || fail 'Unexpected file at API socket path'
rm -f -- "$STATE/api.sock"
printf '%s\n' "$IMAGE_ID" > "$STATE/image-id"
printf '{"release":"R8-candidate","startup_health":"LOADING","quality":"NOT_RUN","tools":"NOT_RUN","512k":"NOT_RUN"}\n' > "$STATE/release-profile.json"
docker run -d --pull never --name "$ENGINE" --init --restart no \
  --label "deepinfer.owner=$OWNER" --label "deepinfer.root=$ROOT" \
  --gpus all --network none --ipc host --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  --read-only --tmpfs /tmp:rw,size=8g "${USER_ARGS[@]}" \
  -e DEEPINFER_R6_TOOL_SUFFIX_GUARD=0 \
  -v "$model_dir:/models/DeepSeek-V4.1-Flash-1017:ro" -v "$STATE:/state" \
  -v "$ROOT/configs/runtime.json:/opt/deepinfer/configs/runtime.json:ro" \
  -v "$ROOT/source/r8-hf2/files/vllm/v1/attention/backends/mla/indexer.py:/opt/r8-venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/indexer.py:ro" \
  -v "$ROOT/source/r8-hf2/files/vllm/models/deepseek_v41/nvidia/specialized_decode/ampere/indexer_metadata.py:/opt/r8-venv/lib/python3.12/site-packages/vllm/models/deepseek_v41/nvidia/specialized_decode/ampere/indexer_metadata.py:ro" \
  --log-opt max-size=100m --log-opt max-file=5 "$IMAGE_ID" >/dev/null
startup_cleanup() {
  local rc=$?
  trap - EXIT
  if ((rc != 0)); then
    "${DOCKER[@]}" logs --timestamps --tail 1500 "$ENGINE" > "$STATE/engine-last-start.log" 2>&1 9>&- || true
  fi
  return "$rc"
}
trap startup_cleanup EXIT
trap 'exit 143' TERM
printf 'R8 model loading. Logs: docker logs --tail 80 --timestamps -f %s\n' "$ENGINE"
deadline=$((SECONDS + ${DEEPINFER_START_TIMEOUT:-7200}))
until docker exec "$ENGINE" /opt/r8-venv/bin/python /opt/deepinfer/scripts/live_acceptance.py --health >/dev/null 2>&1; do
  [[ $(docker inspect --format '{{.State.Running}}' "$ENGINE") == true ]] || fail 'R8 engine exited; see runtime/engine-last-start.log'
  ((SECONDS < deadline)) || fail 'R8 startup timed out; container retained for inspection'
  sleep 5
done
docker run -d --pull never --name "$FRONTEND" --init --restart no \
  --label "deepinfer.owner=$OWNER" --label "deepinfer.root=$ROOT" \
  --network host --read-only --tmpfs /tmp:rw,size=64m "${USER_ARGS[@]}" \
  -v "$STATE:/state:ro" --entrypoint /opt/r8-venv/bin/python "$IMAGE_ID" \
  /opt/deepinfer/scripts/frontend.py >/dev/null
sleep 1
[[ $(docker inspect --format '{{.State.Running}}' "$FRONTEND") == true ]] || fail 'R8 frontend failed; inspect status.sh'
python3 - "$STATE/release-profile.json" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]);r=json.loads(p.read_text());r['startup_health']='READY';p.write_text(json.dumps(r,indent=2)+'\n')
PY
printf 'Ready: http://127.0.0.1:%s/v1\nR8 candidate / 1017 / TP8 / PP1 / C32 / 524288 / 0.94\nQuality and performance are separate manual tests, not startup gates.\n' "$port"
