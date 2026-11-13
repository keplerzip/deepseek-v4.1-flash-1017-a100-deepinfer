"""Hotfix lifecycle tests with CPU-only Docker recording, no GPU execution."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import unittest
from test_r8_workflow import R8WorkflowTests as Fixture

ROOT=Path(__file__).resolve().parents[1]
PATCH=ROOT/'updates/R8-HF2-Indexer-20261113/apply.sh'
BIND='  -v "$ROOT/configs/runtime.json:/opt/deepinfer/configs/runtime.json:ro" \\\n'

class HotfixTests(unittest.TestCase):
    def setUp(self):
        self.f=Fixture(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.root=self.f.root; self.config=self.root/'configs/runtime.json'
        c=json.loads(self.config.read_text()); c['speculative_config']['use_local_argmax_reduction']=True
        self.config.write_text(json.dumps(c,indent=2)+'\n')
        p=self.root/'start.sh'; p.write_text(p.read_text().replace(BIND,''))
        (self.root/'SHA256SUMS').write_text('\n'.join(hashlib.sha256((self.root/n).read_bytes()).hexdigest()+'  '+n for n in ('start.sh','configs/runtime.json'))+'\n')


        import tarfile
        original=ROOT/'source/r8-hf2/before'
        from hf2_changes import CHANGES
        with tarfile.open(self.root/'vendor/upstream-pr66497-source.tar.gz','w:gz') as archive:
            for rel in CHANGES:
                archive.add(original/rel,arcname='vllm-d423120cdc36d4cf8456f43ddfcc654b0c69c5ea/'+rel)

    def apply(self,mode='--no-restart'):
        return subprocess.run(['bash',str(PATCH),str(self.root),mode],env=self.f.env,
                              capture_output=True,text=True,timeout=15)

    def test_config_is_changed_and_really_bound_to_engine(self):
        before=json.loads(self.config.read_text()); r=self.apply()
        self.assertEqual(r.returncode,0,r.stderr)
        after=json.loads(self.config.read_text()); before['speculative_config']['use_local_argmax_reduction']=False
        self.assertEqual(before,after)
        self.assertEqual((self.root/'start.sh').read_text().count(BIND),1)
        for line in (self.root/'SHA256SUMS').read_text().splitlines():
            expected,name=line.split('  ',1)
            self.assertEqual(hashlib.sha256((self.root/name).read_bytes()).hexdigest(),expected)
        state=json.loads((self.root/'runtime/r8-hf2.json').read_text())
        self.assertTrue(json.loads((Path(state['backup'])/'configs/runtime.json').read_text())['speculative_config']['use_local_argmax_reduction'])
        from hf2_changes import CHANGES,patched
        original=ROOT/'source/r8-hf2/before'
        for rel in CHANGES:
            self.assertEqual((self.root/'source/r8-hf2/files'/rel).read_bytes(),patched(rel,(original/rel).read_bytes()))
            self.assertIn(str(Path('source/r8-hf2/files')/rel),(self.root/'start.sh').read_text())
        self.assertFalse(any(x[0] in ('stop','kill','rm','run') for x in self.f.calls()))

    def test_reapply_does_not_duplicate_mount(self):
        self.assertEqual(self.apply().returncode,0)
        first=self.config.read_bytes(); script=(self.root/'start.sh').read_bytes()
        r=self.apply(); self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(first,self.config.read_bytes()); self.assertEqual(script,(self.root/'start.sh').read_bytes())

    def test_foreign_container_is_untouched_and_config_not_changed(self):
        self.f.state.write_text(json.dumps(dict(containers={'dsv411017-deepinfer-r8-engine':dict(owner='foreign',root='/other',running=True)})))
        before=self.config.read_bytes(); r=self.apply('--restart')
        self.assertNotEqual(r.returncode,0); self.assertIn('another installation',r.stderr)
        self.assertEqual(before,self.config.read_bytes())
        self.assertFalse(any(x[0] in ('stop','kill','rm','run') for x in self.f.calls()))

    def test_restart_handles_missing_frontend_and_retains_r7(self):
        self.f.state.write_text(json.dumps(dict(containers={
            'dsv411017-deepinfer-r8-engine':dict(owner='deepseek-1017-deepinfer-sm80-r8',root=str(self.root),running=False),
            'dsv411017-deepinfer-engine':dict(owner='stable',root='/R7',running=False)})))
        r=self.apply('--restart'); self.assertEqual(r.returncode,0,r.stderr)
        log=self.root/'runtime/start-r8-hf2.log'; deadline=time.monotonic()+6
        while time.monotonic()<deadline:
            if log.exists() and 'Ready:' in log.read_text():break
            time.sleep(.02)
        self.assertIn('Ready: http://127.0.0.1:8005/v1',log.read_text())
        calls=self.f.calls()
        for x in calls:
            if x[0] in ('stop','kill','rm'):
                self.assertIn(x[-1],('dsv411017-deepinfer-r8-api','dsv411017-deepinfer-r8-engine'))
        self.assertFalse(any(any(v in a for v in ('check_perf','check_beta','benchmark_beta','acceptance.py --suite')) for x in calls for a in x))
        self.assertIn(str(self.root)+'/configs/runtime.json:/opt/deepinfer/configs/runtime.json:ro',next(x for x in calls if x[0]=='run' and '-d' in x))
        self.assertIn('dsv411017-deepinfer-engine',json.loads(self.f.state.read_text())['containers'])

    def test_bad_pinned_source_fails_before_stopping(self):
        (self.root/'vendor/upstream-pr66497-source.tar.gz').write_bytes(b'invalid archive')
        before=self.config.read_bytes(); r=self.apply('--restart')
        self.assertNotEqual(r.returncode,0)
        self.assertEqual(before,self.config.read_bytes())
        self.assertFalse(any(x[0] in ('stop','kill','rm','run') for x in self.f.calls()))

    def test_unknown_launcher_fails_before_stopping(self):
        (self.root/'start.sh').write_text('#!/bin/bash\necho unsupported\n')
        r=self.apply('--restart'); self.assertNotEqual(r.returncode,0)
        self.assertIn('Unsupported start.sh',r.stderr)
        self.assertFalse(any(x[0] in ('stop','kill','rm','run') for x in self.f.calls()))

if __name__=='__main__':unittest.main()
