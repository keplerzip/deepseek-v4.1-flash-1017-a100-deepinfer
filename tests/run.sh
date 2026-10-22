#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh"
suite=${1:-smoke}
case "$suite" in smoke|benchmark|long-context|full|full-window-c32|diagnosis) ;; *) fail 'Usage: bash tests/run.sh [smoke|benchmark|long-context|full|full-window-c32|diagnosis] [options]' ;; esac
if (($#)); then shift; fi
exec 9>"$STATE/operation.lock"
flock -n 9 || fail 'Another runtime operation is active'
image
owned "$ENGINE" || fail 'Start this installation engine first'
docker exec "$ENGINE" /usr/bin/python3 /opt/deepinfer/scripts/live_acceptance.py --health
[[ ! -e "$STATE/cache-acceptance-active" ]] || fail 'Another acceptance marker exists; inspect the current operation first'
run=${DSV411017_REPORT_ID:-"acceptance-$suite-$(date -u +%Y%m%dT%H%M%S)-$$"}
[[ "$run" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,159}$ ]] || fail 'Invalid DSV411017_REPORT_ID'
mkdir "$STATE/$run" || fail 'Report directory exists; select a new ID'
was_running=false
restore_ingress() {
  rm -f -- "$STATE/cache-acceptance-active"
  if [[ "$was_running" == true ]]; then docker start "$FRONTEND" >/dev/null; fi
}
trap restore_ingress EXIT
if owned "$FRONTEND" && [[ $(docker inspect --format '{{.State.Running}}' "$FRONTEND") == true ]]; then
  was_running=true
  docker stop -t 10 "$FRONTEND" >/dev/null
fi
touch "$STATE/cache-acceptance-active"
cp -- "$STATE/effective-config.json" "$STATE/$run/effective-config.json"
cp -- "$STATE/release-profile.json" "$STATE/$run/release-profile.json"
if [[ "$suite" == diagnosis ]]; then
  docker exec "$ENGINE" /usr/bin/python3 /opt/deepinfer/tests/diagnose_speed.py "$@" \
    --unix-socket /state/api.sock --base-url http://localhost --output "/state/$run/report.json"
else
  docker exec "$ENGINE" /usr/bin/python3 /opt/deepinfer/tests/acceptance.py "$@" \
    --unix-socket /state/api.sock --base-url http://localhost --suite "$suite" --output "/state/$run/report.json"
fi
printf 'Report: %s/%s/report.json\n' "$STATE" "$run"
