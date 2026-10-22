#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
image
docker run --rm --pull never --network host --read-only --tmpfs /tmp:rw,size=1g \
  "${USER_ARGS[@]}" -v "$STATE:/state" --entrypoint /usr/bin/python3 "$IMAGE_ID" \
  /opt/deepinfer/scripts/benchmark_beta.py "$@"
