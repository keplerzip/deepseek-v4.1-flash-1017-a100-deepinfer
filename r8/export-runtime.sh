#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
image
mkdir -p "$ROOT/images" "$ROOT/validation"
docker save --output "$ROOT/images/runtime-image.tar.partial" "$IMAGE_TAG"
python3 "$ROOT/scripts/seal_image.py" --archive "$ROOT/images/runtime-image.tar.partial" \
  --tag "$IMAGE_TAG" --image-id "$IMAGE_ID" --report "$ROOT/validation/saved-image.json" --env "$ROOT/manifests/runtime.env"
mv -- "$ROOT/images/runtime-image.tar.partial" "$ROOT/images/runtime-image.tar"
printf 'Complete offline runtime image exported and all saved blobs verified.\n'
