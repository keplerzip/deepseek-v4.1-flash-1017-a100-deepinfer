#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
for name in "$ENGINE" "$FRONTEND"; do
  if owned "$name"; then
    docker inspect --format '{{.Name}} {{.State.Status}} {{.Config.Image}}' "$name"
    docker logs --tail 15 "$name"
  else printf '%s: not running for this installation\n' "$name"; fi
done
