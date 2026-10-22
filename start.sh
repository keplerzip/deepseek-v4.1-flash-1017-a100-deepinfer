#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
model_dir="$ROOT/models/DeepSeek-V4.1-Flash-1017"
port=8005
profile=aggressive
yoco_keep=256
cpu_kv_gib=0
cpu_kv_policy=arc
eager=0
disabled=()
while (($#)); do
  case "$1" in
    --model-dir) model_dir=${2:?Missing model directory}; shift 2 ;;
    --port) port=${2:?Missing port}; shift 2 ;;
    --eager) eager=1; shift ;;
    --cpu-kv-policy) cpu_kv_policy=${2:?Missing arc or lru}; shift 2 ;;
    --cpu-kv-gib) cpu_kv_gib=${2:?Missing total CPU KV GiB}; shift 2 ;;
    --disable) disabled+=("${2:?Use engram, mla, yoco, topk or mhc}"); shift 2 ;;
    --yoco-keep) yoco_keep=${2:?Missing replay tail}; shift 2 ;;
    --profile) profile=${2:?Use aggressive or reference}; shift 2 ;;
    --help) printf 'Usage: bash start.sh [--model-dir PATH-OR-SYMLINK] [--port 8005] [--profile aggressive|reference] [--cpu-kv-gib 0] [--cpu-kv-policy arc|lru] [--yoco-keep 256] [--eager] [--disable engram|mla|yoco|topk|mhc]\nCPU KV is TOTAL across TP8; 0 disables it. Approximate YOCO FFN stays OFF.\n'; exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done
[[ "$port" =~ ^[0-9]+$ ]] && ((port >= 1024 && port <= 65535)) || fail 'Invalid port'
[[ "$profile" == aggressive || "$profile" == reference ]] || fail 'Profile must be aggressive or reference'
[[ "$cpu_kv_policy" == arc || "$cpu_kv_policy" == lru ]] || fail 'CPU KV policy must be arc or lru'
[[ "$cpu_kv_gib" =~ ^[0-9]+$ ]] && ((cpu_kv_gib <= 768)) || fail 'CPU KV must be 0..768 GiB total'
[[ "$yoco_keep" =~ ^[0-9]+$ ]] && ((yoco_keep >= 128 && yoco_keep < 8192)) || fail 'Replay tail must be 128..8191'
[[ -d "$model_dir" ]] || fail "Model directory/soft link not found: $model_dir"
model_dir=$(realpath -e -- "$model_dir")
[[ "$model_dir" != *:* && "$model_dir" != *,* && "$model_dir" != *$'\n'* ]] || fail 'Unsupported model mount path delimiter'
exec 9>"$STATE/operation.lock"
flock -n 9 || fail 'Another runtime operation is active'
image
if docker inspect "$ENGINE" >/dev/null 2>&1; then
  owned "$ENGINE" || fail 'Engine name belongs to another installation'
  fail 'This adaptation already has a container; inspect status.sh and stop.sh before restarting'
fi
if docker inspect "$FRONTEND" >/dev/null 2>&1; then remove_owned "$FRONTEND"; fi
printf '%s\n' "$model_dir" > "$STATE/model.path"
mapfile -t network_ids < <(docker network ls --filter driver=bridge --format '{{.ID}}')
if ((${#network_ids[@]})); then
  docker network inspect "${network_ids[@]}" > "$STATE/docker-networks.json"
else printf '[]\n' > "$STATE/docker-networks.json"; fi
docker run --rm --pull never --network host --read-only --tmpfs /tmp:rw,size=64m \
  "${USER_ARGS[@]}" -v "$STATE:/state" --entrypoint /usr/bin/python3 "$IMAGE_ID" \
  /opt/deepinfer/scripts/policy.py --networks /state/docker-networks.json --output /state/api-listeners.json --port "$port"
docker run --rm --pull never --network host --read-only --tmpfs /tmp:rw,size=64m \
  "${USER_ARGS[@]}" -v "$STATE:/state" --entrypoint /usr/bin/python3 "$IMAGE_ID" \
  /opt/deepinfer/scripts/check_port.py
docker run --rm --pull never --gpus all --network none --read-only --tmpfs /tmp:rw,size=2g \
  "${USER_ARGS[@]}" -v "$model_dir:/models/DeepSeek-V4.1-Flash-1017:ro" -v "$STATE:/state" \
  --entrypoint /usr/bin/python3 "$IMAGE_ID" /opt/deepinfer/scripts/preflight.py \
  --model-dir /models/DeepSeek-V4.1-Flash-1017 --report /state/preflight.json
docker run --rm --pull never --network none --ipc host --read-only --tmpfs /tmp:rw,size=64m \
  "${USER_ARGS[@]}" -v "$STATE:/state" --entrypoint /usr/bin/python3 "$IMAGE_ID" \
  /opt/deepinfer/scripts/cpu_kv_budget.py --requested-gib "$cpu_kv_gib" --policy "$cpu_kv_policy" --report /state/cpu-kv-plan.json
NUMA_ARGS=()
if ((cpu_kv_gib > 0)); then
  NUMA_ARGS=(--cap-add SYS_NICE)
  docker run --rm --pull never "${NUMA_ARGS[@]}" --network none --read-only --tmpfs /tmp:rw,size=64m \
    "${USER_ARGS[@]}" -v "$STATE:/state" -v "$ROOT/adapter/numa_policy.py:/numa-policy.py:ro" \
    --entrypoint /usr/bin/python3 "$IMAGE_ID" /numa-policy.py
fi
bash "$ROOT/check-gpu.sh"
candidate_flag=1
[[ "$profile" == aggressive ]] || candidate_flag=0
topk_flag=$candidate_flag
mhc_flag=$candidate_flag
mla_flag=$candidate_flag
engram_flag=$candidate_flag
yoco_flag=0
for component in "${disabled[@]}"; do
  case "$component" in
    topk) topk_flag=0 ;;
    mhc) mhc_flag=0 ;;
    mla) mla_flag=0 ;;
    engram) engram_flag=0 ;;
    yoco) yoco_flag=0 ;;
    *) fail "Unknown component: $component" ;;
  esac
done
printf '{"release":"20261022-r3-cpukv","profile":"%s","gpu_validation":"KERNEL_GATES_PASSED_MODEL_ACCEPTANCE_PENDING","approximate_yoco_ffn":%s,"yoco_keep":%s,"eager":%s,"flags":{"topk":%s,"mhc":%s,"mla":%s,"engram":%s,"yoco":%s}}\n' "$profile" "$([[ "$yoco_flag" == 1 ]] && printf true || printf false)" "$yoco_keep" "$eager" "$topk_flag" "$mhc_flag" "$mla_flag" "$engram_flag" "$yoco_flag" > "$STATE/release-profile.json"
[[ ! -e "$STATE/api.sock" || -S "$STATE/api.sock" ]] || fail 'Unexpected file at engine socket path'
rm -f -- "$STATE/api.sock"
check_memory_reserve() {
if ! docker run --rm --pull never --network none --ipc host --read-only --tmpfs /tmp:rw,size=64m \
  "${USER_ARGS[@]}" -v "$STATE:/state" --entrypoint /usr/bin/python3 "$IMAGE_ID" \
  /opt/deepinfer/scripts/cpu_kv_budget.py --verify-reserve --report /state/memory-reserve.json; then
  docker stop -t 90 "$ENGINE" >/dev/null
  fail '500 GiB available reserve was not met; new engine stopped and API not published'
fi
}
printf '%s\n' "$IMAGE_ID" > "$STATE/image-id"
touch "$STATE/cache-acceptance-active"
docker run -d --pull never --name "$ENGINE" --init --restart no \
  --label "deepinfer.owner=$OWNER" --label "deepinfer.root=$ROOT" \
  --gpus all --network none --ipc host --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  "${NUMA_ARGS[@]}" --read-only --tmpfs /tmp:rw,size=8g "${USER_ARGS[@]}" \
  -e "DEEPINFER_BETA_UNIFIED_MLA=$mla_flag" -e "DEEPINFER_BETA_ENGRAM_PREFETCH=$engram_flag" \
  -e VLLM_USE_SIMPLE_KV_OFFLOAD=0 -e "DEEPINFER_BETA_EAGER=$eager" -e "DEEPINFER_BETA_YOCO_KEEP=$yoco_keep" -e "DEEPINFER_BETA_YOCO_FFN=$yoco_flag" -e "VLLM_DSV41_DEEPINFER_FUSED_COMBINE=$topk_flag" -e "VLLM_DSV41_DEEPINFER_MHC_ALIGNED=$mhc_flag" \
  -v "$model_dir:/models/DeepSeek-V4.1-Flash-1017:ro" -v "$STATE:/state" \
  -v "$ROOT/scripts/launch.py:/opt/deepinfer/scripts/launch.py:ro" \
  -v "$ROOT/scripts/benchmark_beta.py:/opt/deepinfer/scripts/benchmark_beta.py:ro" \
  -v "$ROOT/configs/runtime.json:/opt/deepinfer/configs/runtime.json:ro" \
  -v "$ROOT/adapter/official_server.py:/opt/deepinfer/adapter/official_server.py:ro" \
  -v "$ROOT/adapter/native_kv_offload.py:/opt/deepinfer/adapter/native_kv_offload.py:ro" \
  -v "$ROOT/adapter/rank_major_offload.py:/opt/deepinfer/adapter/rank_major_offload.py:ro" \
  -v "$ROOT/adapter/numa_policy.py:/opt/deepinfer/adapter/numa_policy.py:ro" \
  --log-opt max-size=100m --log-opt max-file=5 "$IMAGE_ID" >/dev/null
( exec "${DOCKER[@]}" logs --follow --timestamps "$ENGINE" ) &
engine_log_pid=$!
trap 'kill "$engine_log_pid" 2>/dev/null || true; wait "$engine_log_pid" 2>/dev/null || true; rm -f -- "$STATE/cache-acceptance-active"; docker logs --timestamps "$ENGINE" > "$STATE/engine-last-start.log" 2>&1 || true' EXIT
trap 'exit 143' TERM
deadline=$((SECONDS + ${DEEPINFER_START_TIMEOUT:-7200}))
until docker exec "$ENGINE" /usr/bin/python3 /opt/deepinfer/scripts/live_acceptance.py --health >/dev/null 2>&1; do
  [[ $(docker inspect --format '{{.State.Running}}' "$ENGINE") == true ]] || fail 'Engine exited; see runtime/engine-last-start.log'
  ((SECONDS < deadline)) || fail 'Startup timed out; engine retained for inspection, API not published'
  sleep 5
done
check_memory_reserve
docker exec "$ENGINE" /usr/bin/python3 /opt/deepinfer/scripts/live_acceptance.py --report /state/live-acceptance.json
docker exec "$ENGINE" /usr/bin/python3 /opt/deepinfer/scripts/kv_acceptance.py --report /state/cpu-kv-acceptance.json
docker exec "$ENGINE" /usr/bin/python3 /opt/deepinfer/scripts/benchmark_beta.py --uds /state/api.sock --url http://localhost --requests 32 --concurrency 32 --prompt-tokens 8192 --report /state/c32-acceptance.json
docker exec "$ENGINE" /usr/bin/python3 /opt/deepinfer/scripts/benchmark_beta.py --uds /state/api.sock --url http://localhost --requests 1 --concurrency 1 --prompt-tokens 520192 --output-tokens 128 --report /state/long-context-acceptance.json
docker exec "$ENGINE" /usr/bin/python3 /opt/deepinfer/tests/acceptance.py --unix-socket /state/api.sock --suite full --warmups 2 --single-samples 10 --load-requests 128 --output-tokens 1024 --output /state/full-acceptance.json
docker exec "$ENGINE" /usr/bin/python3 /opt/deepinfer/tests/diagnose_speed.py --unix-socket /state/api.sock --warmups 2 --samples 10 --output-tokens 1024 --output /state/speed-diagnosis.json
rm -f -- "$STATE/cache-acceptance-active"
check_memory_reserve
docker exec "$ENGINE" /usr/bin/python3 /opt/deepinfer/scripts/finalize_acceptance.py
docker run -d --pull never --name "$FRONTEND" --init --restart no \
  --label "deepinfer.owner=$OWNER" --label "deepinfer.root=$ROOT" \
  --network host --read-only --tmpfs /tmp:rw,size=64m "${USER_ARGS[@]}" \
  -v "$STATE:/state:ro" --entrypoint /usr/bin/python3 "$IMAGE_ID" \
  /opt/deepinfer/scripts/frontend.py >/dev/null
sleep 1
[[ $(docker inspect --format '{{.State.Running}}' "$FRONTEND") == true ]] || fail 'Frontend did not bind; inspect status.sh (port may already be used)'
printf 'Ready: http://127.0.0.1:%s/v1\nModel: DeepSeek-Flash\nTP8 / EP8 / C32 / 524288 / DSpark k5\nCompute profile: %s\nEvidence: runtime/preflight.json, release-profile.json, live-acceptance.json\n' "$port" "$profile"
