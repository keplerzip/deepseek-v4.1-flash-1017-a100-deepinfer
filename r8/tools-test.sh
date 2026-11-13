#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
image
docker run --rm --pull never --runtime runc --network host --read-only \
  --tmpfs /tmp:rw,size=64m "${USER_ARGS[@]}" \
  -v "$ROOT:/opt/deepinfer:ro" -v "$STATE:/state" \
  --entrypoint /opt/r8-venv/bin/python "$IMAGE_ID" /opt/deepinfer/scripts/probe_tools.py \
  --base-url http://127.0.0.1:8005/v1 "$@"
