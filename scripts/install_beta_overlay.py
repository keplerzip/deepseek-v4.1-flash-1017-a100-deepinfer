"""Install experimental files only over the exact, pinned R2 source bytes."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path('/opt/deepinfer')
source = root / 'source/beta-overlay'
target = Path('/usr/local/lib/python3.12/dist-packages')
manifest = json.loads((source / 'manifest.json').read_text())
for item in manifest['files']:
    for path, expected in [(target / item['path'], item['base_sha256']),
                           (source / 'files' / item['path'], item['beta_sha256']),
                           (source / item['patch'], item['patch_sha256'])]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Source checksum differs: {path}')
for item in manifest['files']:
    shutil.copy2(source / 'files' / item['path'], target / item['path'])
shutil.copy2(root / 'source/beta_runtime.py', target / 'vllm/models/deepseek_v4_1/beta_runtime.py')
