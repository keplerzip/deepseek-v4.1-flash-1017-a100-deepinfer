#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
image
# R3-HF1: explicit per-device graph capture streams in the bundled GPU gates.
GPU_GATE_BINDS=(-v "$ROOT/scripts/check_beta_kernels.py:/opt/deepinfer/scripts/check_beta_kernels.py:ro"
                -v "$ROOT/scripts/check_sm80_candidate.py:/opt/deepinfer/scripts/check_sm80_candidate.py:ro")
docker run --rm --pull never --gpus all --network none --read-only --tmpfs /tmp:rw,size=4g \
  "${USER_ARGS[@]}" "${GPU_GATE_BINDS[@]}" -v "$STATE:/state" --entrypoint /usr/bin/python3 "$IMAGE_ID" \
  /opt/deepinfer/scripts/check_beta_kernels.py --mode gpu --report /state/beta-mla-gpu.json
docker run --rm --pull never --gpus all --network none --read-only --tmpfs /tmp:rw,size=4g \
  "${USER_ARGS[@]}" "${GPU_GATE_BINDS[@]}" -v "$STATE:/state" --entrypoint /usr/bin/python3 "$IMAGE_ID" \
  /opt/deepinfer/scripts/check_sm80_candidate.py --mode gpu --image-id "$IMAGE_ID" \
  --package-root /usr/local/lib/python3.12/dist-packages --report /state/selected-sm80-gpu.json
