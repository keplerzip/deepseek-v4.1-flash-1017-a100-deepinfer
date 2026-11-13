"""Apply sealed metadata-only local corrections after the unmodified PR overlay."""
import hashlib
import json
from pathlib import Path
import shutil
import sysconfig

root=Path(__file__).resolve().parents[1]/'source/r8-hf2'
manifest=json.loads((root/'manifest.json').read_text())
site=Path(sysconfig.get_paths()['purelib'])
pending=[]
for row in manifest['files']:
    source=root/'files'/row['path']; target=site/row['path']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==row['sha256'],row['path']
    assert hashlib.sha256(target.read_bytes()).hexdigest()==row['before_sha256'],row['path']
    pending.append((source,target))
for source,target in pending:shutil.copy2(source,target)
print(json.dumps(dict(status='PASS',update=manifest['update'],installed_files=len(pending),gpu_executed=False)))
