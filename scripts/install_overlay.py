"""Build-time reconstruction of reviewed changes against immutable R2 bytes."""
import importlib.util
import json
from pathlib import Path
import shutil

root = Path('/opt/deepinfer')
spec = importlib.util.spec_from_file_location('selected_overlay', root / 'source/sm80-overlay/apply.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
package_root = Path('/usr/local/lib/python3.12/dist-packages')
generated = Path('/tmp/deepinfer-generated-overlay')
report = module.prepare(package_root, generated)
for entry in report['files']:
    shutil.copy2(generated / entry['path'], package_root / entry['path'])
(root / 'validation').mkdir(exist_ok=True)
(root / 'validation/build-overlay.json').write_text(json.dumps(report, indent=2) + '\n')
shutil.rmtree(generated)
