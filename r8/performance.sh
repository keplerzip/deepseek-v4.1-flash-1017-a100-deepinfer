#!/usr/bin/env bash
# Serving-only benchmark/diagnosis. Keep the existing model and API running.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
[[ $# == 0 ]] || fail 'Usage: bash performance.sh'
exec 9>"$STATE/operation.lock"
flock -n 9 || fail 'Another runtime operation is active'
image
owned "$ENGINE" || fail 'Start this installation engine first'
[[ $(docker inspect --format '{{.State.Running}}' "$ENGINE") == true ]] || fail 'Engine is not running'
[[ $(docker inspect --format '{{.Image}}' "$ENGINE") == "$IMAGE_ID" ]] || fail 'Running engine image differs from this package'
owned "$FRONTEND" || fail 'Owned frontend not found'
[[ $(docker inspect --format '{{.State.Running}}' "$FRONTEND") == true ]] || fail 'Frontend is not running'
[[ -f "$STATE/image-id" && $(cat "$STATE/image-id") == "$IMAGE_ID" ]] || fail 'Startup image identity mismatch'
marker="$STATE/cache-acceptance-active"
[[ ! -e "$marker" && ! -L "$marker" ]] || fail 'An acceptance marker already exists; inspect the active operation'
run="perf-only-$(date -u +%Y%m%dT%H%M%S)-$$"
report_dir="$STATE/$run"
mkdir "$report_dir"
# Creation is exclusive; cleanup checks its contents before removing our marker.
(set -o noclobber; printf '%s\n' "$run" > "$marker") || fail 'Could not own the acceptance marker'
stage=setup
child_pid=
child_pgid=
benchmark_rc=-1
diagnosis_rc=-1
run_child() {
  set -m
  ( trap - EXIT INT TERM; "$@" ) 9>&- &
  child_pid=$!
  child_pgid=$child_pid
  local rc=0
  wait "$child_pid" || rc=$?
  set +m
  child_pid=
  child_pgid=
  return "$rc"
}
cancel_client() {
  local phase=$1
  [[ -f "$report_dir/active-$phase.pid" ]] || return 0
  local -a cancel_prefix
  if [[ "${DOCKER[0]}" == sudo ]]; then
    cancel_prefix=(sudo -n timeout --kill-after=2s 20s docker)
  else
    cancel_prefix=(timeout --kill-after=2s 20s "${DOCKER[@]}")
  fi
  command "${cancel_prefix[@]}" exec "$ENGINE" /opt/r8-venv/bin/python -c '
import os, pathlib, signal, sys, time
path, run, phase = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
try: pid = int(path.read_text().strip())
except (FileNotFoundError, ValueError): sys.exit(1)
def ours():
    try:
        env = pathlib.Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        return (("DSV411017_PERF_RUN=" + run).encode() in env and
                ("DSV411017_PERF_PHASE=" + phase).encode() in env)
    except (FileNotFoundError, ProcessLookupError): return False
if ours():
    os.kill(pid, signal.SIGTERM)
    until = time.monotonic() + 10
    while ours() and time.monotonic() < until: time.sleep(.2)
    if ours(): os.kill(pid, signal.SIGKILL)
    until = time.monotonic() + 5
    while ours() and time.monotonic() < until: time.sleep(.2)
    if ours(): sys.exit(1)
' "/state/$run/active-$phase.pid" "$run" "$phase" 9>&-
}
run_engine() {
  local phase=$1 rc=0
  shift
  run_child "${DOCKER[@]}" exec -e "DSV411017_PERF_RUN=$run" \
    -e "DSV411017_PERF_PHASE=$phase" "$ENGINE" /bin/sh -c \
    'printf "%s\n" "$$" > "/state/$DSV411017_PERF_RUN/active-$DSV411017_PERF_PHASE.pid"; exec /opt/r8-venv/bin/python "$@"' \
    sh "$@" || rc=$?
  if ((rc == 0)); then
    rm -f -- "$report_dir/active-$phase.pid"
  elif cancel_client "$phase"; then
    rm -f -- "$report_dir/active-$phase.pid"
  else
    printf 'ERROR: Could not confirm cancellation of the %s client.\n' "$phase" >&2
  fi
  return "$rc"
}
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  # Cancel only run+phase-marked Python clients inside the existing engine.
  # Doing this first normally lets docker exec (including sudo) return itself.
  for phase in benchmark diagnosis; do
    if ! cancel_client "$phase"; then
      printf 'ERROR: Could not confirm cancellation of our %s client; model/API were not stopped.\n' "$phase" >&2
      rc=1
    fi
  done
  if [[ -n "$child_pid" ]]; then
    child_alive() {
      kill -0 "$child_pid" 2>/dev/null && return 0
      if [[ "${DOCKER[0]}" == sudo ]]; then
        command sudo -n kill -0 -- "-$child_pgid" 2>/dev/null
      else
        kill -0 -- "-$child_pgid" 2>/dev/null
      fi
    }
    signal_child_group() {
      local sig=$1
      if [[ "${DOCKER[0]}" == sudo ]]; then
        command sudo -n kill "-$sig" -- "-$child_pgid" 2>/dev/null || true
      else
        kill "-$sig" -- "-$child_pgid" 2>/dev/null || true
      fi
    }
    for _ in {1..20}; do child_alive || break; sleep .1; done
    if child_alive; then signal_child_group TERM; fi
    for _ in {1..20}; do child_alive || break; sleep .1; done
    if child_alive; then signal_child_group KILL; fi
    for _ in {1..20}; do child_alive || break; sleep .1; done
    if child_alive; then
      printf 'ERROR: Our local exec client did not exit; cleanup did not wait indefinitely.\n' >&2
      rc=1
    else
      wait "$child_pid" 2>/dev/null || true
    fi
  fi
  if [[ -f "$marker" && $(cat -- "$marker" 2>/dev/null) == "$run" ]]; then
    rm -f -- "$marker"
  fi
  python3 - "$report_dir/run-status.json" "$run" "$IMAGE_ID" "$rc" "$benchmark_rc" "$diagnosis_rc" "$stage" <<'PY'
import json, pathlib, sys
path, run, image, rc, benchmark, diagnosis, stage = sys.argv[1:]
rc, benchmark, diagnosis = int(rc), int(benchmark), int(diagnosis)
row = {'status':'PASS' if rc == benchmark == diagnosis == 0 else 'FAIL',
       'exit_code':rc, 'run':run, 'image_id':image, 'stage':stage,
       'benchmark_exit_code':None if benchmark < 0 else benchmark,
       'diagnosis_exit_code':None if diagnosis < 0 else diagnosis,
       'scope':'serving benchmark and diagnosis only; API remains available',
       'gpu_kernel_validation':'NOT_RUN', 'protocol_validation':'NOT_RUN',
       'long_context_512k_validation':'NOT_RUN', 'api_stop_requested':False}
pathlib.Path(path).write_text(json.dumps(row, indent=2)+'\n')
PY
  printf 'Performance exit status: %s\nReports: %s\n' "$rc" "$report_dir"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for name in effective-config.json release-profile.json image-id; do
  [[ ! -f "$STATE/$name" ]] || cp -- "$STATE/$name" "$report_dir/$name"
done
printf 'API remains available. Running benchmark and diagnosis on the loaded engine.\nOther requests are detected by accounting and may make the result contaminated.\nReports: %s\n' "$report_dir"
stage=benchmark
benchmark_rc=0
run_engine benchmark /opt/deepinfer/tests/acceptance.py --unix-socket /state/api.sock \
  --base-url http://localhost --suite benchmark --warmups 2 --single-samples 10 \
  --load-requests 128 --output-tokens 1024 --output "/state/$run/benchmark.json" || benchmark_rc=$?
printf 'Benchmark exit status: %s; continuing with diagnosis.\n' "$benchmark_rc"
stage=diagnosis
diagnosis_rc=0
run_engine diagnosis /opt/deepinfer/tests/diagnose_speed.py --unix-socket /state/api.sock \
  --base-url http://localhost --warmups 2 --samples 10 --output-tokens 1024 \
  --output "/state/$run/diagnosis.json" || diagnosis_rc=$?
stage=complete
if ((benchmark_rc != 0 || diagnosis_rc != 0)); then exit 1; fi
# A zero process exit without its new PASS report is not successful evidence.
python3 - "$report_dir/benchmark.json" "$report_dir/diagnosis.json" <<'PY'
import json, pathlib, sys
for name in sys.argv[1:]:
    report=json.loads(pathlib.Path(name).read_text())
    if report.get('status') != 'PASS':
        raise SystemExit('Performance client did not produce a PASS report: '+name)
PY
