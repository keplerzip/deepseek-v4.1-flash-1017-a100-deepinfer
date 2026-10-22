#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
STATE="$ROOT/runtime"
ENGINE=dsv411017-deepinfer-engine
FRONTEND=dsv411017-deepinfer-api
OWNER=deepseek-1017-deepinfer-sm80-formal
BASE_IMAGE_ID=sha256:a63405a72ce202bb8347f17fc2432be0df484b794fba0224fd210b6daff004f6
umask 077
mkdir -p "$STATE" "$STATE/home" "$STATE/cache" "$STATE/tmp"
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
if command docker info >/dev/null 2>&1; then
  DOCKER=(docker)
elif command sudo -n docker info >/dev/null 2>&1; then
  DOCKER=(sudo -n docker)
else
  fail 'Docker access is required (docker group or sudo -n docker)'
fi
docker() { command "${DOCKER[@]}" "$@"; }
owned() { [[ $(docker inspect --format '{{index .Config.Labels "deepinfer.owner"}} {{index .Config.Labels "deepinfer.root"}}' "$1" 2>/dev/null || true) == "$OWNER $ROOT" ]]; }
image() {
  [[ -f "$ROOT/manifests/runtime.env" ]] || fail 'Run install.sh after copying the complete runtime package'
  source "$ROOT/manifests/runtime.env"
  [[ "$IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] || fail 'Invalid runtime image identity'
  local actual
  actual=$(docker image inspect --format '{{.Id}}' "$IMAGE_TAG" 2>/dev/null || true)
  local accepted=false allowed
  for allowed in ${IMAGE_ALLOWED_IDS:-$IMAGE_ID}; do
    [[ "$allowed" =~ ^sha256:[0-9a-f]{64}$ ]] || fail 'Invalid accepted image identity'
    [[ "$actual" != "$allowed" ]] || accepted=true
  done
  [[ "$accepted" == true ]] || fail 'The pinned DeepInfer adaptation image is missing or changed; run install.sh'
  # Docker's classic store reports the config digest; containerd can report
  # the OCI index. Both are checked against the saved archive's descriptors.
  IMAGE_ID=$actual
}
USER_ARGS=(--user "$(id -u):$(id -g)" --workdir /state
  -e HOME=/state/home -e TMPDIR=/state/tmp -e XDG_CACHE_HOME=/state/cache
  -e HF_HOME=/state/cache/hf -e TRITON_CACHE_DIR=/state/cache/triton
  -e TORCHINDUCTOR_CACHE_DIR=/state/cache/inductor -e TORCH_EXTENSIONS_DIR=/state/cache/torch-extensions
  -e CUDA_CACHE_PATH=/state/cache/cuda -e HTTP_PROXY= -e HTTPS_PROXY= -e ALL_PROXY=
  -e http_proxy= -e https_proxy= -e all_proxy=)
for group_id in $(id -G); do USER_ARGS+=(--group-add "$group_id"); done
container_python() {
  docker run --rm --pull never --network none --read-only --tmpfs /tmp:rw,size=2g \
    "${USER_ARGS[@]}" -v "$ROOT:/package:ro" -v "$STATE:/state" \
    --entrypoint /usr/bin/python3 "$IMAGE_ID" "$@"
}
remove_owned() {
  if docker inspect "$1" >/dev/null 2>&1; then
    owned "$1" || fail "Container belongs to another installation: $1"
    docker stop -t 90 "$1" >/dev/null
    docker rm "$1" >/dev/null
  fi
}
