"""Check the published R8 source/evidence boundaries; no model execution."""
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
R8=ROOT/'r8'
subprocess.run(['sha256sum','--check','--quiet','SHA256SUMS'],cwd=R8,check=True)
for source_manifest in ('r8-pr66497','r8-hf2'):
    folder=R8/'source'/source_manifest
    manifest=json.loads((folder/'manifest.json').read_text())
    for row in manifest['files']:
        source=folder/'files'/row['path']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==row['sha256'],source
        before=folder/'before'/row['path']
        if before.exists():
            assert hashlib.sha256(before.read_bytes()).hexdigest()==row['before_sha256'],before
config=json.loads((R8/'configs/runtime.json').read_text())
assert config['max_model_len']==524288 and config['gpu_memory_utilization']==.94
assert config['max_num_seqs']==32 and config['enable_expert_parallel'] is False
assert config['speculative_config']['num_speculative_tokens']==5
assert config['speculative_config']['use_local_argmax_reduction'] is False
record=json.loads((ROOT/'validation/target-r8-hf2-performance-20261113.json').read_text())
assert record['status']==record['benchmark_status']=='FAIL'
assert record['diagnosis_status']=='CONTAMINATED'
assert record['source']['raw_target_reports_independently_retrieved'] is False
summary=ROOT/record['source']['summary_path']
assert hashlib.sha256(summary.read_bytes()).hexdigest()==record['source']['summary_sha256']
assert record['contamination']['off']['excess_tokens']==7225
assert record['contamination']['global_accounting']['excess_tokens']==45458
for path in ('README.md','docs/r8-hf2-performance-20261113.md'):
    doc=(ROOT/path).read_text()
    assert 'FAIL' in doc and 'CONTAMINATED' in doc
    for row in record['benchmarks'].values():
        for value in row.values():assert f'{value:.2f}' in doc
    for row in record['diagnosis_summary'].values():
        for key in ('decode_tps_median','ttft_seconds_median'):
            assert f'{row[key]:.2f}' in doc
provenance=json.loads((R8/'SOURCE-PROVENANCE.json').read_text())
assert provenance['upstream_head']=='d423120cdc36d4cf8456f43ddfcc654b0c69c5ea'
assert provenance['upstream_base']=='ace415c06892a6119c88c8c57ed6fd761b722e32'
assert provenance['pinned_pr_parent']=='d33c17cb78ef59a6242891cebc3dddd7b5e2301c'
server=Path('vendor/official-recipe/source/server-py/main.py')
assert (R8/server).read_bytes()==(ROOT/server).read_bytes()
command=(R8/'updates/R8-HF2-one-command.txt').read_text().strip()
code=command.split("python3 -c '",1)[1][:-1]
tree=ast.parse(code)
encoded=ast.literal_eval(tree.body[1].value.args[0].args[0])
decoded=gzip.decompress(base64.b64decode(encoded))
assert decoded==(R8/'updates/R8-HF2-Indexer-20261113/apply.sh').read_bytes()
assert hashlib.sha256(decoded).hexdigest()=='c7deb003d1bf400137253e392d1e52a77895ca13b2fe451b9cfca9b627fe2090'
ack=(ROOT/'ACKNOWLEDGMENTS.md').read_text()
for marker in ('yaoyaoding','66497','HalfVulpes/Deepinfer-SM80','deepseek-recipe','wtdcode/vllm-backport'):
    assert marker in ack
print('PASS: R8 fixed sources, original contaminated statuses, protocol provenance, base64 payload and acknowledgments; GPU NOT_RUN')
