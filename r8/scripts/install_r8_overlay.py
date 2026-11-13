"""Install the exact Python-only PR onto its matching upstream binary wheel."""
import hashlib
import json
from pathlib import Path
import shutil
import sysconfig

root = Path(__file__).resolve().parents[1]
snapshot = root / 'source/r8-pr66497'
manifest = json.loads((snapshot / 'manifest.json').read_text())
site = Path(sysconfig.get_paths()['purelib'])
pending = []
for row in manifest['files']:
    source = snapshot / 'files' / row['path']
    if hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
        raise RuntimeError('PR source mismatch: ' + row['path'])
    if row['kind'] != 'runtime':
        continue
    target = site / row['path']
    actual = hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
    if actual != row['base_sha256']:
        raise RuntimeError('Upstream wheel differs from PR base: ' + row['path'])
    pending.append((source, target))
for source, target in pending:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
print(json.dumps({'status': 'PASS', 'head': manifest['head'],
                  'installed_runtime_files': len(pending), 'gpu_executed': False}))
