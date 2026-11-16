"""Validate the current R9 public tree and operator-reported performance."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
files=[p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts
       and '__pycache__' not in p.relative_to(ROOT).parts]
for p in files:
    rel=p.relative_to(ROOT)
    if rel.parts[0] in {'runtime','models','images','dist','precompiled'}:
        raise AssertionError('Local artifact in public tree: '+str(rel))
    if p.suffix=='.py':ast.parse(p.read_text(),filename=str(rel))
    if p.suffix=='.sh':subprocess.run(['bash','-n',str(p)],check=True)
    if p.suffix=='.json':json.loads(p.read_text())
    assert p.stat().st_size<=50*1024**2,rel
    assert p.suffix not in {'.safetensors','.gguf','.pt','.pth','.so','.cubin','.ptx','.pem','.key'},rel
    if p.name!='tokenizer.json' and p.suffix!='.whl':
        try:text=p.read_text()
        except UnicodeDecodeError:continue
        assert not re.search(r'gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----',text),rel
subprocess.run(['sha256sum','--check','--quiet','SHA256SUMS'],cwd=ROOT,check=True)
for name in ('r8-pr66497','r8-hf2','r9-overlay'):
    folder=ROOT/'source'/name
    for row in json.loads((folder/'manifest.json').read_text())['files']:
        assert hashlib.sha256((folder/'files'/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
record=json.loads((ROOT/'validation/target-r9-performance-20261116.json').read_text())
assert record['benchmark_status']==record['diagnosis_status']=='PASS'
assert record['contamination']==[]
assert record['source']['raw_target_reports_independently_retrieved'] is False
assert record['diagnosis_summary'] is None and not record['diagnosis_details_provided']
summary=ROOT/record['source']['summary_path']
assert hashlib.sha256(summary.read_bytes()).hexdigest()==record['source']['summary_sha256']
expected={'off':(363.73,.20,2253.57,16.69),'high':(271.61,.20,1841.59,19.71)}
for mode,values in expected.items():
    assert tuple(record['benchmarks'][mode].values())==values
for name in ('README.md','docs/performance.md'):
    text=(ROOT/name).read_text()
    assert 'PASS' in text and '[]' in text
    for values in expected.values():
        for value in values:assert f'{value:.2f}' in text,(name,value)
assert not list((ROOT/'validation').glob('target-r8-*'))
assert not list((ROOT/'validation').glob('target-hf5-*'))
assert not (ROOT/'r8').exists()
config=json.loads((ROOT/'configs/runtime.json').read_text())
assert config['max_model_len']==524288 and config['max_num_seqs']==32
assert config['gpu_memory_utilization']==.94 and config['enable_expert_parallel'] is False
assert config['speculative_config']['num_speculative_tokens']==5
assert config['speculative_config']['use_local_argmax_reduction'] is False
assert config['engram_config']['cpu_offload'] is True
assert config['swa_bounded_replay'] is False
provenance=json.loads((ROOT/'vendor/official-recipe/provenance.json').read_text())
server=ROOT/'vendor/official-recipe'/provenance['server_path']
assert hashlib.sha256(server.read_bytes()).hexdigest()==provenance['server_sha256']
for path in (ROOT/'docs').glob('*.md'):
    assert 'R9' in path.read_text(),path
for name in ['README.md']+[str(p.relative_to(ROOT)) for p in (ROOT/'docs').glob('*.md')]:
    text=(ROOT/name).read_text()
    for link in re.findall(r'\]\(([^)]+)\)',text):
        if '://' in link or link.startswith('#'):continue
        assert ((ROOT/name).parent/link.split('#')[0]).exists(),(name,link)
ack=(ROOT/'ACKNOWLEDGMENTS.md').read_text()
for author in ('yaoyaoding','tomasruizt','WoosukKwon','StevenWang-CY','HalfVulpes','wtdcode'):
    assert author in ack,author
perf=(ROOT/'performance.sh').read_text()
assert '--single-samples 10' in perf and '--load-requests 128' in perf and '--output-tokens 1024' in perf
assert 'check_perf_kernels.py' not in perf and 'check_beta_kernels.py' not in perf
assert '--suite benchmark' in perf and 'API remains available' in perf
print(f'PASS: {len(files)} R9 source files/assets, latest PASS report, provenance and publication boundary; no GPU execution')
