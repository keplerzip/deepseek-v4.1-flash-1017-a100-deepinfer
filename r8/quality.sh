#!/usr/bin/env bash
# CPU HTTP client only; no second CUDA context and no API admission change.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
image
exec 9>"$STATE/operation.lock"
flock -n 9 || fail 'Another R8 quality client is running'
docker run --rm --pull never --runtime runc --network host --read-only \
  --tmpfs /tmp:rw,size=1g "${USER_ARGS[@]}" \
  -v "$ROOT:/opt/deepinfer:ro" -v "$STATE:/state" \
  --entrypoint /opt/r8-venv/bin/python "$IMAGE_ID" /opt/deepinfer/scripts/quality_suite.py "$@"
