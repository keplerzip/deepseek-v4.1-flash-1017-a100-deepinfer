#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
exec 9>"$STATE/operation.lock"
flock -n 9 || fail 'Another runtime operation is active'
remove_owned "$FRONTEND"
remove_owned "$ENGINE"
