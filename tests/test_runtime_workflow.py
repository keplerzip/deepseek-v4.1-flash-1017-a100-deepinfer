"""Source-package shell integration using a fake Docker CLI, never GPU evidence."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT if (ROOT / "start.sh").is_file() else Path("/package")
IMAGE = "sha256:" + "b" * 64
BASE = "sha256:a63405a72ce202bb8347f17fc2432be0df484b794fba0224fd210b6daff004f6"

DOCKER_STUB = r'''
import json, os, sys
from pathlib import Path
a=sys.argv[1:]
log=Path(os.environ['WORKFLOW_STUB_LOG'])
with log.open('a') as f: f.write(json.dumps(a)+'\n')
path=Path(os.environ['WORKFLOW_STUB_STATE'])
state=json.loads(path.read_text()) if path.exists() else {'containers':{}}
def done(code=0,out=''):
 path.write_text(json.dumps(state))
 if out: print(out)
 raise SystemExit(code)
if a[0]=='info': done()
if a[:2]==['image','inspect']:
 tag=a[-1]
 if tag=='deepseek-v4.1-flash-1017-a100:20261018-r2': done(out=os.environ['WORKFLOW_BASE_IMAGE'])
 done(out=os.environ['WORKFLOW_IMAGE'])
if a[:2]==['network','ls']: done()
if a[0]=='inspect':
 name=a[-1]
 c=state['containers'].get(name)
 if not c: done(1)
 fmt=a[a.index('--format')+1] if '--format' in a else ''
 if '.Config.Labels' in fmt: done(out=c.get('owner','foreign')+' '+c.get('root','/foreign'))
 if '.State.Running' in fmt: done(out='true')
 done(out='{}')
if a[0]=='run':
 if any('cpu_kv_budget.py' in item for item in a) and '--verify-reserve' in a: done(int(os.environ.get('WORKFLOW_RAM_EXIT','0')))
 if any('check_port.py' in item for item in a): done(int(os.environ.get('WORKFLOW_PORT_EXIT','0')))
 if any('preflight.py' in item for item in a): done(int(os.environ.get('WORKFLOW_PREFLIGHT_EXIT','0')))
 if any('check_sm80_candidate.py' in item for item in a): done(int(os.environ.get('WORKFLOW_GPU_GATE_EXIT','0')))
 if '-d' in a:
  name=a[a.index('--name')+1]
  labels=[a[i+1] for i,item in enumerate(a) if item=='--label']
  labels=dict(item.split('=',1) for item in labels)
  state['containers'][name]={'owner':labels.get('deepinfer.owner'),'root':labels.get('deepinfer.root')}
  done(out='fixture-container-id')
 done()
if a[0]=='exec':
 if '--health' in a: done()
 done(int(os.environ.get('WORKFLOW_ACCEPTANCE_EXIT','0')))
if a[0] in ('stop','start'): done()
if a[0]=='rm':
 state['containers'].pop(a[-1],None)
 done()
if a[0]=='logs': done(out='Fixture Docker log; no model loaded')
if a[0] in ('load','build'): done()
done(1)
'''


@unittest.skipUnless((SOURCE / "start.sh").is_file(), "Source shell package is not mounted; these are host workflow tests")
class RuntimeWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.root = self.directory / "package with spaces"
        (self.root / "scripts").mkdir(parents=True)
        (self.root / "manifests").mkdir()
        for name in ("start.sh", "stop.sh", "status.sh", "install.sh", "build-runtime.sh", "check-gpu.sh"):
            shutil.copy2(SOURCE / name, self.root / name)
        shutil.copy2(SOURCE / "scripts/common.sh", self.root / "scripts/common.sh")
        (self.root / "tests").mkdir()
        shutil.copy2(SOURCE / "tests/run.sh", self.root / "tests/run.sh")
        (self.root / "manifests/runtime.env").write_text(f"IMAGE_TAG=candidate:fixture\nIMAGE_ID={IMAGE}\n")
        model = self.directory / "existing 1017 model"
        model.mkdir()
        (self.root / "models").mkdir()
        (self.root / "models/DeepSeek-V4.1-Flash-1017").symlink_to(model, target_is_directory=True)
        self.model = model
        binary = self.directory / "bin"
        binary.mkdir()
        docker = binary / "docker"
        docker.write_text("#!" + sys.executable + "\n" + DOCKER_STUB)
        docker.chmod(0o755)
        sleep = binary / "sleep"
        sleep.write_text("#!/bin/sh\nexit 0\n")
        sleep.chmod(0o755)
        bootstrap = self.directory / "stub-command.bash"
        bootstrap.write_text('''# Test-only interception; /tmp may be mounted noexec.
command() {
  if [[ "${1:-}" == docker ]]; then
    shift
    "$WORKFLOW_STUB_PYTHON" "$WORKFLOW_STUB_DOCKER" "$@"
  else
    builtin command "$@"
  fi
}
sleep() { :; }
''')
        self.log = self.directory / "docker.log"
        self.state = self.directory / "state.json"
        self.env = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ.get("PATH", ""),
                    "WORKFLOW_STUB_LOG": str(self.log), "WORKFLOW_STUB_STATE": str(self.state),
                    "WORKFLOW_IMAGE": IMAGE, "WORKFLOW_BASE_IMAGE": BASE,
                    "BASH_ENV": str(bootstrap), "WORKFLOW_STUB_PYTHON": sys.executable,
                    "WORKFLOW_STUB_DOCKER": str(docker)}

    def run_script(self, name, *args, environment=None):
        return subprocess.run(["bash", str(self.root / name), *args], env={**self.env, **(environment or {})},
                              text=True, capture_output=True, timeout=20)

    def calls(self):
        return [json.loads(row) for row in self.log.read_text().splitlines()] if self.log.exists() else []

    def launches(self):
        return [row for row in self.calls() if row[0] == "run" and "-d" in row]

    def test_start_orders_gpu_gate_and_live_acceptance_before_publication(self):
        result = self.run_script("start.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertTrue(any("check_sm80_candidate.py" in item for row in calls for item in row))
        engine = next(i for i, row in enumerate(calls) if row[0] == "run" and "-d" in row)
        acceptance = next(i for i, row in enumerate(calls) if row[0] == "exec" and "--report" in row)
        frontend = max(i for i, row in enumerate(calls) if row[0] == "run" and "-d" in row)
        self.assertLess(engine, acceptance)
        self.assertLess(acceptance, frontend)
        engine_args = calls[engine]
        self.assertIn("VLLM_DSV41_DEEPINFER_FUSED_COMBINE=1", engine_args)
        self.assertIn("VLLM_DSV41_DEEPINFER_MHC_ALIGNED=1", engine_args)
        self.assertIn("DEEPINFER_BETA_YOCO_FFN=0", engine_args)
        policy = next(row for row in calls if any('policy.py' in item for item in row))
        self.assertEqual(policy[policy.index('--port') + 1], '8005')
        self.assertIn(str(self.model.resolve()) + ":/models/DeepSeek-V4.1-Flash-1017:ro", engine_args)
        self.assertIn("none", engine_args)
        self.assertIn("--read-only", engine_args)
        self.assertIn("--user", engine_args)

    def test_cpu_kv_numa_probe_and_engine_capability_do_not_change_frontend(self):
        result = self.run_script('start.sh', '--cpu-kv-gib', '768')
        self.assertEqual(result.returncode, 0, result.stderr)
        calls=self.calls()
        probe=next(i for i,row in enumerate(calls) if '/numa-policy.py' in row)
        engine=next(i for i,row in enumerate(calls) if row[0]=='run' and '-d' in row)
        self.assertLess(probe,engine)
        self.assertIn('SYS_NICE',calls[probe])
        self.assertIn('SYS_NICE',calls[engine])
        self.assertNotIn('SYS_NICE',self.launches()[-1])
        self.assertTrue(any('/adapter/rank_major_offload.py:' in arg for arg in calls[engine]))

    def test_default_disables_cpu_kv_and_mounts_requested_runtime(self):
        result = self.run_script('start.sh')
        self.assertEqual(result.returncode, 0, result.stderr)
        calls=self.calls()
        budget=next(row for row in calls if any('cpu_kv_budget.py' in arg for arg in row) and '--requested-gib' in row)
        self.assertEqual(budget[budget.index('--requested-gib')+1], '0')
        self.assertFalse(any('/numa-policy.py' in row for row in calls))
        engine=self.launches()[0]
        self.assertNotIn('SYS_NICE', engine)
        self.assertIn(str(self.root)+'/scripts/benchmark_beta.py:/opt/deepinfer/scripts/benchmark_beta.py:ro',engine)
        self.assertIn(str(self.root)+'/configs/runtime.json:/opt/deepinfer/configs/runtime.json:ro',engine)
        self.assertIn(str(self.root)+'/adapter/official_server.py:/opt/deepinfer/adapter/official_server.py:ro',engine)

    def test_aggressive_profile_publishes_explicit_unvalidated_marker(self):
        result = self.run_script("start.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        marker = json.loads((self.root / "runtime/release-profile.json").read_text())
        self.assertFalse(marker["approximate_yoco_ffn"])
        self.assertEqual(marker["gpu_validation"], "KERNEL_GATES_PASSED_MODEL_ACCEPTANCE_PENDING")
        for flag in ("DEEPINFER_BETA_UNIFIED_MLA=1", "DEEPINFER_BETA_ENGRAM_PREFETCH=1", "DEEPINFER_BETA_YOCO_FFN=0"):
            self.assertIn(flag, self.launches()[0])

    def test_formal_rejects_approximate_yoco_opt_in(self):
        result = self.run_script('start.sh', '--enable-yoco')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.launches(), [])

    def test_disable_and_eager_controls_are_passed_to_engine(self):
        result = self.run_script('start.sh', '--disable', 'engram', '--disable', 'yoco', '--eager')
        self.assertEqual(result.returncode, 0, result.stderr)
        engine = self.launches()[0]
        self.assertIn('DEEPINFER_BETA_ENGRAM_PREFETCH=0', engine)
        self.assertIn('DEEPINFER_BETA_YOCO_FFN=0', engine)
        self.assertIn('DEEPINFER_BETA_UNIFIED_MLA=1', engine)
        self.assertIn('DEEPINFER_BETA_EAGER=1', engine)
        marker = json.loads((self.root / 'runtime/release-profile.json').read_text())
        self.assertFalse(marker['approximate_yoco_ffn'])
        self.assertEqual(marker['flags']['engram'], 0)
        self.assertEqual(marker['eager'], 1)

    def test_failed_gpu_gate_never_starts_model(self):
        result = self.run_script('start.sh', environment={'WORKFLOW_GPU_GATE_EXIT':'1'})
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.launches(), [])

    def test_low_available_ram_stops_new_engine_without_publishing_api(self):
        result = self.run_script('start.sh', environment={'WORKFLOW_RAM_EXIT':'1'})
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.launches()), 1)
        self.assertTrue(any(row[0]=='stop' and row[-1]=='dsv411017-deepinfer-engine' for row in self.calls()))

    def test_formal_runs_cache_capacity_and_reserve_gates_before_publication(self):
        result = self.run_script('start.sh')
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        front = max(i for i,row in enumerate(calls) if row[0]=='run' and '-d' in row)
        for script in ('kv_acceptance.py','benchmark_beta.py','cpu_kv_budget.py','finalize_acceptance.py','/tests/acceptance.py','/tests/diagnose_speed.py'):
            indexes = [i for i,row in enumerate(calls) if any(script in x for x in row)]
            self.assertTrue(indexes, script)
            self.assertLess(max(indexes), front)

    def test_independent_acceptance_pauses_and_restores_only_owned_ingress_even_on_failure(self):
        result = self.run_script('start.sh')
        self.assertEqual(result.returncode, 0, result.stderr)
        (self.root/'runtime/effective-config.json').write_text('{}')
        result = self.run_script('tests/run.sh', 'benchmark', environment={'WORKFLOW_ACCEPTANCE_EXIT':'1'})
        self.assertNotEqual(result.returncode, 0)
        calls=self.calls()
        stop=next(i for i,row in enumerate(calls) if row[0]=='stop' and row[-1]=='dsv411017-deepinfer-api')
        test=next(i for i,row in enumerate(calls) if '/opt/deepinfer/tests/acceptance.py' in row and '--suite' in row and row[row.index('--suite')+1]=='benchmark')
        restored=max(i for i,row in enumerate(calls) if row[0]=='start' and row[-1]=='dsv411017-deepinfer-api')
        self.assertLess(stop,test)
        self.assertLess(test,restored)
        self.assertFalse((self.root/'runtime/cache-acceptance-active').exists())

    def test_failed_preflight_never_runs_candidate_or_engine(self):
        result = self.run_script("start.sh", environment={"WORKFLOW_PREFLIGHT_EXIT": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.launches(), [])
        self.assertFalse(any("check_sm80_candidate.py" in item for row in self.calls() for item in row))

    def test_port_collision_fails_before_any_gpu_allocation(self):
        result = self.run_script("start.sh", environment={"WORKFLOW_PORT_EXIT": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any("--gpus" in row for row in self.calls()))
        self.assertEqual(self.launches(), [])

    def test_actual_port_checker_preserves_occupied_listener(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            policy = self.directory / "policy.json"
            policy.write_text(json.dumps({"schema": 1, "authentication": "disabled", "port": port,
                "listeners": [{"address": "127.0.0.1", "port": port, "kind": "loopback",
                               "allowed_sources": ["127.0.0.1/32"]}]}))
            result = subprocess.run([sys.executable, str(SOURCE / "scripts/check_port.py"), "--policy", str(policy)],
                                    capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no service was stopped", result.stderr)
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                pass

    def test_off_leaves_both_candidates_disabled(self):
        result = self.run_script("start.sh", "--profile", "reference")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(any("check_sm80_candidate.py" in item for row in self.calls() for item in row))
        engine = self.launches()[0]
        self.assertIn("VLLM_DSV41_DEEPINFER_FUSED_COMBINE=0", engine)
        self.assertIn("VLLM_DSV41_DEEPINFER_MHC_ALIGNED=0", engine)

    def test_failed_live_acceptance_is_retained_and_api_not_published(self):
        result = self.run_script('start.sh', environment={'WORKFLOW_ACCEPTANCE_EXIT':'1'})
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.launches()), 1)
        self.assertIn('dsv411017-deepinfer-engine', self.launches()[0])

    def test_foreign_engine_is_not_stopped_or_removed(self):
        self.state.write_text(json.dumps({"containers": {"dsv411017-deepinfer-engine": {"owner": "foreign", "root": "/foreign"}}}))
        result = self.run_script("start.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.launches(), [])
        self.assertFalse(any(row[0] in ("stop", "rm") for row in self.calls()))

    def test_operation_lock_prevents_concurrent_start(self):
        (self.root / "runtime").mkdir()
        with (self.root / "runtime/operation.lock").open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_script("start.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("operation", result.stderr)
        self.assertEqual(self.launches(), [])

    def test_build_rejects_wrong_base_image_before_docker_build(self):
        result = self.run_script("build-runtime.sh", environment={"WORKFLOW_BASE_IMAGE": "sha256:" + "c" * 64})
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(row[0] == "build" for row in self.calls()))

    def test_install_checks_package_checksum_before_loading_any_image(self):
        (self.root / "SHA256SUMS").write_text("0" * 64 + "  start.sh\n")
        result = self.run_script("install.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(row[0] in ("load", "run") for row in self.calls()))

    def test_public_install_requires_image_when_archive_is_absent(self):
        digest=hashlib.sha256((self.root/'start.sh').read_bytes()).hexdigest()
        (self.root/'SHA256SUMS').write_text(digest+'  start.sh\n')
        result=self.run_script('install.sh',environment={'WORKFLOW_IMAGE':'sha256:'+'c'*64})
        self.assertNotEqual(result.returncode,0)
        self.assertIn('public Git source does not include',result.stderr)
        self.assertFalse(any(row[0] in ('load','run') for row in self.calls()))

    def test_public_install_checks_current_source_with_existing_image(self):
        digest=hashlib.sha256((self.root/'start.sh').read_bytes()).hexdigest()
        (self.root/'SHA256SUMS').write_text(digest+'  start.sh\n')
        result=self.run_script('install.sh')
        self.assertEqual(result.returncode,0,result.stderr)
        checks=next(row for row in self.calls() if row[0]=='run')
        self.assertIn(str(self.root)+':/opt/deepinfer:ro',checks)
        self.assertFalse(any(row[0]=='load' for row in self.calls()))

    def test_launcher_uses_actual_pinned_parser_request_logging_flag(self):
        launcher = (SOURCE / "scripts/launch.py").read_text()
        # The exact R2 make_arg_parser was exercised separately in its image:
        # this version has BooleanOptionalAction on enable_log_requests.
        self.assertNotIn("--disable-log-requests", launcher)
        self.assertIn("--no-enable-log-requests", launcher)


if __name__ == "__main__":
    unittest.main()
