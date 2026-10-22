#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
tag=deepseek-v4.1-flash-1017-a100-deepinfer:20261022-r3-cpukv
[[ $(docker image inspect --format '{{.Id}}' deepseek-v4.1-flash-1017-a100:20261018-r2) == "$BASE_IMAGE_ID" ]] || fail 'Pinned R2 base image identity differs'
mkdir -p "$ROOT/validation" "$ROOT/manifests" "$ROOT/images"
docker build --network none --pull=false --progress plain -t "$tag" "$ROOT" 2>&1 | tee "$ROOT/validation/image-build.log"
actual=$(docker image inspect --format '{{.Id}}' "$tag")
printf 'IMAGE_TAG=%s\nIMAGE_ID=%s\n' "$tag" "$actual" > "$ROOT/manifests/runtime.env"
docker image inspect "$tag" > "$ROOT/validation/image-inspect.json"
printf 'Built %s\n' "$actual"
