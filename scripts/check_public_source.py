"""No-GPU public source checks. This does not benchmark or validate a model."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.git','runtime','images','dist','__pycache__'}
files=[p for p in ROOT.rglob('*') if p.is_file() and not any(part in EXCLUDED for part in p.relative_to(ROOT).parts)]
for path in files:
    if path.suffix=='.py':ast.parse(path.read_text(),filename=str(path))
    if path.suffix=='.sh':subprocess.run(['bash','-n',str(path)],check=True)
    if path.suffix=='.json':json.loads(path.read_text())
    if path.stat().st_size>50*1024**2:raise AssertionError('Oversized source asset: '+str(path.relative_to(ROOT)))
    if path.suffix in {'.safetensors','.gguf','.pt','.pth','.pem','.key'}:raise AssertionError('Excluded publication asset: '+str(path.relative_to(ROOT)))
    if path.name!='tokenizer.json' and path.suffix!='.whl':
        try:text=path.read_text()
        except UnicodeDecodeError:continue
        if re.search(r'gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----',text):
            raise AssertionError('Possible credential in '+str(path.relative_to(ROOT)))
subprocess.run(['sha256sum','--check','--quiet','SHA256SUMS'],cwd=ROOT,check=True)
provenance=json.loads((ROOT/'vendor/official-recipe/provenance.json').read_text())
server=ROOT/'vendor/official-recipe'/provenance['server_path']
assert hashlib.sha256(server.read_bytes()).hexdigest()==provenance['server_sha256']
record=json.loads((ROOT/'validation/target-hf5-acceptance-20261023.json').read_text())
assert record['deployment_context']['cpu_kv_gib_total']==0
assert record['deployment_context']['gpu_memory_utilization']==0.95
for name in ('hostname','installation_path'):assert name not in record['deployment_context']
summary=ROOT/record['source']['summary_path']
assert hashlib.sha256(summary.read_bytes()).hexdigest()==record['source']['summary_sha256']
readme=(ROOT/'README.md').read_text()
for row in record['benchmarks'].values():
    for value in row.values():assert f'{value:.2f}' in readme
for row in record['diagnosis_summary'].values():
    for key in ('decode_tps_median','ttft_seconds_median'):assert f'{row[key]:.2f}' in readme
config=json.loads((ROOT/'configs/runtime.json').read_text())
assert config['gpu_memory_utilization']==.95 and config['enable_prefix_caching']
assert config['max_model_len']==524288 and config['max_num_seqs']==32
assert 'cpu_kv_gib=0' in (ROOT/'start.sh').read_text()
print(f'PASS: {len(files)} source/assets checked; GPU/model/performance NOT_RUN by this check')
