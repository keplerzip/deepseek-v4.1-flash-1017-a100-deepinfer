#!/usr/bin/env bash
# Explicit real-model qualification, separate from startup and performance.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
label=R8
base=http://127.0.0.1:8005/v1
gsm=200
baseline=
while (($#)); do
  case "$1" in
    --label) label=${2:?}; shift 2 ;;
    --base-url) base=${2:?}; shift 2 ;;
    --gsm-limit) gsm=${2:?}; shift 2 ;;
    --baseline) baseline=${2:?Use /state/quality-R7.json}; shift 2 ;;
    *) fail 'Usage: bash acceptance.sh [--label R7|R8] [--base-url URL/v1] [--gsm-limit 200|1319] [--baseline /state/quality-R7.json]' ;;
  esac
done
[[ "$label" =~ ^[A-Za-z0-9_-]+$ ]] || fail 'Invalid report label'
image
exec 9>"$STATE/operation.lock"
flock -n 9 || fail 'Another R8 quality client is running'
client() {
  docker run --rm --pull never --runtime runc --network host --read-only --tmpfs /tmp:rw,size=1g \
    "${USER_ARGS[@]}" -v "$ROOT:/opt/deepinfer:ro" -v "$STATE:/state" \
    --entrypoint /opt/r8-venv/bin/python "$IMAGE_ID" "$@"
}
rc=0
client /opt/deepinfer/scripts/live_acceptance.py --base-url "${base%/v1}" \
  --report "/state/smoke-$label.json" || rc=1
client /opt/deepinfer/scripts/probe_tools.py --base-url "$base" \
  --output "/state/tools-$label.json" || rc=1
client /opt/deepinfer/scripts/probe_tools.py --base-url "$base" --reasoning-effort high --max-tokens 4096 --tool-choice auto --timeout 300 \
  --output "/state/tools-high-$label.json" || rc=1
client /opt/deepinfer/scripts/quality_suite.py --base-url "$base" --label "$label" \
  --gsm-limit "$gsm" --long-context --output "/state/quality-$label.json" || rc=1
if [[ -n "$baseline" ]]; then
  client /opt/deepinfer/scripts/quality_suite.py --compare "$baseline" "/state/quality-$label.json" \
    --output "/state/compare-$label.json" || rc=1
fi
python3 - "$STATE" "$label" "$baseline" "$rc" <<'PY' || rc=1
import json,pathlib,sys
root,label,baseline,rc=sys.argv[1:];p=pathlib.Path(root);issues=[]
for name in ['smoke','tools','tools-high']:
    try:r=json.loads((p/f'{name}-{label}.json').read_text())
    except (OSError,ValueError):r={}
    if r.get('status')!='PASS':issues.append(name+' not passed')
try:q=json.loads((p/f'quality-{label}.json').read_text())
except (OSError,ValueError):q={}
if q.get('status')!='MEASURED' or q.get('long_context_512k')!='PASS':issues.append('quality/512K incomplete or failed')
if baseline:
    try:c=json.loads((p/f'compare-{label}.json').read_text())
    except (OSError,ValueError):c={}
    if c.get('status')!='PASS':issues.append('paired comparison needs review')
elif label!='R7':issues.append('No R7 paired baseline supplied; standalone measurement only')
r=dict(status=('PASS' if baseline else 'MEASURED') if not issues and rc=='0' else 'REVIEW_REQUIRED',issues=issues,label=label,
       scope='Real HTTP smoke, tools, sampled quality and 512K; finite tests do not establish broad model equivalence')
(p/f'acceptance-{label}.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
raise SystemExit(0 if r['status'] in ('PASS','MEASURED') else 1)
PY
printf 'Reports: %s/*-%s.json\nAPI remains available; no model/container was stopped.\n' "$STATE" "$label"
exit "$rc"
