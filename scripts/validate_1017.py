"""Fast 1017 metadata identity check; does not read weights or touch any GPU."""
import hashlib
import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
model=Path(sys.argv[1])
manifest=json.loads((root/'vendor/model-manifest.json').read_text())
for name in ('config.json','tokenizer.json','model.safetensors.index.json'):
    row=next(r for r in manifest['files'] if r['path']==name)
    p=model/name
    if not p.is_file() or p.stat().st_size!=row['size'] or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
        raise SystemExit('1017 metadata identity differs: '+name)
print('1017 config/tokenizer/weight index match; no weight scan or device checks.')
