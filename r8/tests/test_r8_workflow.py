"""R8 shell lifecycle integration with a CPU-only Docker recorder."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from fake_docker import DOCKER_STUB

ROOT=Path(__file__).resolve().parents[1]
IMAGE='sha256:'+'b'*64

class R8WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        d=Path(self.temp.name);self.root=d/'R8 with spaces';self.root.mkdir()
        for name in ('start.sh','stop.sh','performance.sh'):shutil.copy2(ROOT/name,self.root/name)
        for name in ('scripts','manifests','models','vendor','configs'):(self.root/name).mkdir()
        shutil.copy2(ROOT/'configs/runtime.json',self.root/'configs/runtime.json')
        for name in ('common.sh','validate_1017.py'):shutil.copy2(ROOT/'scripts'/name,self.root/'scripts'/name)
        (self.root/'manifests/runtime.env').write_text('IMAGE_TAG=test:r8\nIMAGE_ID='+IMAGE+'\n')
        m=self.root/'models/DeepSeek-V4.1-Flash-1017';m.mkdir();rows=[]
        for name in ('config.json','tokenizer.json','model.safetensors.index.json'):
            data=b'{"cpu_test_fixture":true}\n';(m/name).write_bytes(data)
            rows.append(dict(path=name,size=len(data),sha256=hashlib.sha256(data).hexdigest()))
        (self.root/'vendor/model-manifest.json').write_text(json.dumps(dict(files=rows)))
        binary=d/'bin';binary.mkdir();docker=binary/'docker'
        docker.write_text('#!'+sys.executable+'\n'+DOCKER_STUB);docker.chmod(0o755)
        bootstrap=d/'bootstrap.bash';bootstrap.write_text('sleep() { :; }\ncommand() {\n  if [[ "${1:-}" == docker ]]; then\n    shift\n    "$WORKFLOW_STUB_PYTHON" "$WORKFLOW_STUB_DOCKER" "$@"\n  else\n    builtin command "$@"\n  fi\n}\n')
        self.state=d/'state.json';self.log=d/'docker.log'
        self.env=dict(os.environ,PATH=str(binary)+':'+os.environ['PATH'],BASH_ENV=str(bootstrap),
            WORKFLOW_STUB_LOG=str(self.log),WORKFLOW_STUB_STATE=str(self.state),
            WORKFLOW_IMAGE=IMAGE,WORKFLOW_BASE_IMAGE=IMAGE,WORKFLOW_ROOT=str(self.root),
            WORKFLOW_STUB_PYTHON=sys.executable,WORKFLOW_STUB_DOCKER=str(docker))

    def run_shell(self,name,**env):
        return subprocess.run(['bash',str(self.root/name)],env=self.env|env,capture_output=True,text=True,timeout=15)

    def calls(self):return [json.loads(x) for x in self.log.read_text().splitlines()]

    def test_stable_running_is_untouched(self):
        self.state.write_text(json.dumps(dict(containers={'dsv411017-deepinfer-engine':dict(running=True,owner='stable',root='/R7')})))
        r=self.run_shell('start.sh');self.assertNotEqual(r.returncode,0);self.assertIn('R8 did not stop it',r.stderr)
        self.assertFalse(any(x[0] in ('stop','kill','rm','run') for x in self.calls()))

    def test_start_publishes_after_health_without_test_gates(self):
        r=self.run_shell('start.sh');self.assertEqual(r.returncode,0,r.stderr)
        calls=self.calls();launch=[x for x in calls if x[0]=='run' and '-d' in x]
        self.assertEqual([x[x.index('--name')+1] for x in launch],['dsv411017-deepinfer-r8-engine','dsv411017-deepinfer-r8-api'])
        self.assertLess(calls.index(launch[0]),next(i for i,x in enumerate(calls) if x[0]=='exec' and '--health' in x))
        self.assertFalse(any(any(y in arg for y in ('check_perf','check_beta','preflight.py','acceptance.py --suite','benchmark_beta','r7_compute')) for x in calls for arg in x))
        self.assertIn('8005',r.stdout)
        self.assertIn(str(self.root)+'/configs/runtime.json:/opt/deepinfer/configs/runtime.json:ro',launch[0])

    def test_busy_port_does_not_launch_or_stop_containers(self):
        r=self.run_shell('start.sh',WORKFLOW_PORT_EXIT='1');self.assertNotEqual(r.returncode,0)
        self.assertFalse(any(x[0] in ('stop','rm','kill') or (x[0]=='run' and '-d' in x) for x in self.calls()))

    def test_stop_only_removes_owned_r8_containers(self):
        state=dict(containers={n:dict(owner='deepseek-1017-deepinfer-sm80-r8',root=str(self.root),running=True)
            for n in ('dsv411017-deepinfer-r8-engine','dsv411017-deepinfer-r8-api')})
        state['containers']['dsv411017-deepinfer-engine']=dict(owner='R7',root='/stable',running=True)
        self.state.write_text(json.dumps(state));r=self.run_shell('stop.sh');self.assertEqual(r.returncode,0,r.stderr)
        remaining=json.loads(self.state.read_text())['containers']
        self.assertEqual(list(remaining),['dsv411017-deepinfer-engine'])

    def test_performance_keeps_api_and_skips_gpu_operator_clients(self):
        self.assertEqual(self.run_shell('start.sh').returncode,0)
        r=self.run_shell('performance.sh');self.assertEqual(r.returncode,0,r.stderr)
        calls=self.calls();self.assertFalse(any(x[0] in ('stop','kill') for x in calls))
        self.assertFalse(any('check_perf_kernels.py' in arg or 'check_beta_kernels.py' in arg for x in calls for arg in x))
        self.assertTrue(any('/opt/deepinfer/tests/acceptance.py' in x for x in calls))

if __name__=='__main__':unittest.main()
