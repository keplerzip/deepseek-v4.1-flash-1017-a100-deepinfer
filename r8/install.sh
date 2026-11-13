#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
cd "$ROOT"
if [[ "${1:-}" == --verify ]]; then
  sha256sum --check --quiet SHA256SUMS
elif [[ $# != 0 ]]; then
  fail 'Usage: bash install.sh [--verify]'
fi
if ! (image) >/dev/null 2>&1; then
  [[ -f "$ROOT/images/runtime-image.tar" ]] || fail 'Complete offline image missing'
  docker load -i "$ROOT/images/runtime-image.tar"
fi
image
printf 'R8 offline runtime installed. No GPU/device/operator/benchmark checks run.\nUse: bash start.sh --model-dir /path/to/DeepSeek-V4.1-Flash-1017\nDefault port 8005; stable R7 containers remain untouched.\n'
