"""Install the sealed R9 backport onto the exact R8 runtime image."""
import hashlib
import json
from pathlib import Path
import shutil
import sysconfig

root=Path(__file__).resolve().parents[1]/'source/r9-overlay'
manifest=json.loads((root/'manifest.json').read_text())
site=Path(sysconfig.get_paths()['purelib'])
pending=[]
for row in manifest['files']:
    source=root/'files'/row['path']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==row['sha256'],row['path']
    if row['kind']!='runtime':continue
    target=site/row['path']
    actual=hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
    assert actual==row['before_sha256'],('Unexpected R8 base',row['path'],actual)
    pending.append((source,target))
for source,target in pending:
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
print(json.dumps(dict(status='PASS',release=manifest['release'],installed_files=len(pending),gpu_executed=False)))
