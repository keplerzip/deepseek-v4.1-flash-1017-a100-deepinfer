#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
[[ -f "$ROOT/SHA256SUMS" ]] || fail 'Package checksum manifest is missing'
cd "$ROOT"
sha256sum --check --quiet SHA256SUMS
if ! (image) >/dev/null 2>&1; then
  [[ -f "$ROOT/images/runtime-image.tar" ]] || fail 'Pinned runtime image is required; public Git source does not include images/runtime-image.tar. See README.'
  docker load -i "$ROOT/images/runtime-image.tar"
fi
image
docker run --rm --pull never --network none --read-only --tmpfs /tmp:rw,size=3g \
  "${USER_ARGS[@]}" -v "$ROOT:/package:ro" -v "$ROOT:/opt/deepinfer:ro" -v "$STATE:/state" \
  --entrypoint /usr/bin/python3 "$IMAGE_ID" /opt/deepinfer/scripts/cpu_checks.py
printf 'Pinned runtime present and public-source CPU checks passed.\nUse: bash start.sh --model-dir /path/to/DeepSeek-V4.1-Flash-1017 --cpu-kv-gib 0\n'
