#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
tag=deepseek-v4.1-flash-1017-a100-deepinfer:20261113-r8-pr66497-8005
mkdir -p "$ROOT/validation" "$ROOT/manifests"
docker build --network none --pull=false --progress plain -f "$ROOT/Dockerfile.r8" -t "$tag" "$ROOT" 2>&1 | tee "$ROOT/validation/image-build.log"
actual=$(docker image inspect --format '{{.Id}}' "$tag")
printf 'IMAGE_TAG=%s\nIMAGE_ID=%s\n' "$tag" "$actual" > "$ROOT/manifests/runtime.env"
