#!/usr/bin/env bash
set -euo pipefail
NEW=${1:-/ai/services/deepseek-v4.1-flash-1017-a100-deepinfer-r8/deepseek-v4.1-flash-1017-a100-deepinfer-r8}
MODE=${2:---restart}
[[ "$MODE" == --restart || "$MODE" == --no-restart ]] || { echo 'Use: bash apply.sh INSTALL_DIR [--restart|--no-restart]' >&2; exit 1; }
[[ -f "$NEW/scripts/common.sh" && -f "$NEW/configs/runtime.json" ]] || { echo "R8 installation not found: $NEW" >&2; exit 1; }
source "$NEW/scripts/common.sh"
[[ "$OWNER" == deepseek-1017-deepinfer-sm80-r8 && "$ENGINE" == dsv411017-deepinfer-r8-engine && "$FRONTEND" == dsv411017-deepinfer-r8-api ]] || fail 'This hotfix requires the isolated R8 installation'
exec 7>"$STATE/r8-hf2.lock"
flock -n 7 || fail 'Another R8 HF2 patch is active'
python3 - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
"""Narrow metadata dispatch correction; leave the stock hardware guard intact."""
import hashlib

CHANGES={
    'vllm/v1/attention/backends/mla/indexer.py': {
        'before_sha256':'2c21e5d414c0c5b04ece8adf96168888da28196aa7e8236a37d50e15a826870e',
        'edits':[
            ('        self.indexer_uses_fp4 = dsa_indexer_uses_fp4(self.vllm_config)\n',
             '        self.indexer_uses_fp4 = self._resolve_indexer_uses_fp4()\n'),
            ('    def _dcp_localize_decode_seq_lens(\n',
             '    def _resolve_indexer_uses_fp4(self) -> bool:\n'
             '        """Resolve the cache format for this metadata backend."""\n'
             '        return dsa_indexer_uses_fp4(self.vllm_config)\n\n'
             '    def _dcp_localize_decode_seq_lens(\n')],
    },
    'vllm/models/deepseek_v41/nvidia/specialized_decode/ampere/indexer_metadata.py': {
        'before_sha256':'7d6f40e6d8e6c9438817d9d800c7a51c1430d4eacfba6de91fbc699b9882d685',
        'edits':[
            ('from . import indexer_uses_fp4\n','from . import indexer_uses_fp4, is_ampere\n'),
            ('    def __init__(self, *args, **kwargs) -> None:\n',
             '    def _resolve_indexer_uses_fp4(self) -> bool:\n'
             '        """Use SM80 codecs, without enabling the stock Blackwell kernels."""\n'
             '        if not is_ampere():\n'
             '            raise ValueError("Ampere indexer metadata requires an SM80 GPU")\n'
             '        return indexer_uses_fp4(self.vllm_config)\n\n'
             '    def __init__(self, *args, **kwargs) -> None:\n')],
    },
}

def patched(path,data):
    row=CHANGES[path]
    if hashlib.sha256(data).hexdigest()!=row['before_sha256']:
        raise RuntimeError('Unexpected pinned source: '+path)
    text=data.decode()
    for old,new in row['edits']:
        if text.count(old)!=1:raise RuntimeError('Ambiguous patch anchor: '+path)
        text=text.replace(old,new)
    compile(text,path,'exec')
    return text.encode()


def sources_to_patch(root):
    import tarfile
    prefix='vllm-d423120cdc36d4cf8456f43ddfcc654b0c69c5ea/'
    with tarfile.open(root/'vendor/upstream-pr66497-source.tar.gz','r:gz') as archive:
        return {rel:patched(rel,archive.extractfile(prefix+rel).read()) for rel in CHANGES}

r=Path(sys.argv[1]); sources_to_patch(r); c=json.loads((r/'configs/runtime.json').read_text())
assert c['speculative_config']['method']=='dspark' and c['speculative_config']['num_speculative_tokens']==5, 'Unexpected draft configuration'
assert c['model']=='/models/DeepSeek-V4.1-Flash-1017', 'Unexpected model'
s=(r/'start.sh').read_text()
assert s.count('  -v "$model_dir:/models/DeepSeek-V4.1-Flash-1017:ro" -v "$STATE:/state" \\\n')==1, 'Unsupported start.sh; no changes made'
PY
for name in "$FRONTEND" "$ENGINE"; do
  if docker inspect "$name" >/dev/null 2>&1; then
    owned "$name" || fail "Container belongs to another installation: $name"
  fi
done
if [[ "$MODE" == --restart ]]; then
  for name in "$FRONTEND" "$ENGINE"; do
    if docker inspect "$name" >/dev/null 2>&1; then
      docker logs --timestamps --tail 1500 "$name" > "$STATE/hf2-before-$name.log" 2>&1 || true
      docker stop -t 30 "$name" >/dev/null
    fi
  done
fi
exec 8>"$STATE/operation.lock"
flock -w 45 8 || fail 'The previous R8 operation still holds its lock; no lock file was removed. Retry after it exits.'
python3 - "$ROOT" <<'PY'
import hashlib,json,os,shutil,stat,sys,tempfile,time
from pathlib import Path
"""Narrow metadata dispatch correction; leave the stock hardware guard intact."""
import hashlib

CHANGES={
    'vllm/v1/attention/backends/mla/indexer.py': {
        'before_sha256':'2c21e5d414c0c5b04ece8adf96168888da28196aa7e8236a37d50e15a826870e',
        'edits':[
            ('        self.indexer_uses_fp4 = dsa_indexer_uses_fp4(self.vllm_config)\n',
             '        self.indexer_uses_fp4 = self._resolve_indexer_uses_fp4()\n'),
            ('    def _dcp_localize_decode_seq_lens(\n',
             '    def _resolve_indexer_uses_fp4(self) -> bool:\n'
             '        """Resolve the cache format for this metadata backend."""\n'
             '        return dsa_indexer_uses_fp4(self.vllm_config)\n\n'
             '    def _dcp_localize_decode_seq_lens(\n')],
    },
    'vllm/models/deepseek_v41/nvidia/specialized_decode/ampere/indexer_metadata.py': {
        'before_sha256':'7d6f40e6d8e6c9438817d9d800c7a51c1430d4eacfba6de91fbc699b9882d685',
        'edits':[
            ('from . import indexer_uses_fp4\n','from . import indexer_uses_fp4, is_ampere\n'),
            ('    def __init__(self, *args, **kwargs) -> None:\n',
             '    def _resolve_indexer_uses_fp4(self) -> bool:\n'
             '        """Use SM80 codecs, without enabling the stock Blackwell kernels."""\n'
             '        if not is_ampere():\n'
             '            raise ValueError("Ampere indexer metadata requires an SM80 GPU")\n'
             '        return indexer_uses_fp4(self.vllm_config)\n\n'
             '    def __init__(self, *args, **kwargs) -> None:\n')],
    },
}

def patched(path,data):
    row=CHANGES[path]
    if hashlib.sha256(data).hexdigest()!=row['before_sha256']:
        raise RuntimeError('Unexpected pinned source: '+path)
    text=data.decode()
    for old,new in row['edits']:
        if text.count(old)!=1:raise RuntimeError('Ambiguous patch anchor: '+path)
        text=text.replace(old,new)
    compile(text,path,'exec')
    return text.encode()


def sources_to_patch(root):
    import tarfile
    prefix='vllm-d423120cdc36d4cf8456f43ddfcc654b0c69c5ea/'
    with tarfile.open(root/'vendor/upstream-pr66497-source.tar.gz','r:gz') as archive:
        return {rel:patched(rel,archive.extractfile(prefix+rel).read()) for rel in CHANGES}

r=Path(sys.argv[1]); cp=r/'configs/runtime.json'; sp=r/'start.sh'
c=json.loads(cp.read_text()); c['speculative_config']['use_local_argmax_reduction']=False
s=sp.read_text(); binding='  -v "$ROOT/configs/runtime.json:/opt/deepinfer/configs/runtime.json:ro" \\\n'
anchor='  -v "$model_dir:/models/DeepSeek-V4.1-Flash-1017:ro" -v "$STATE:/state" \\\n'
if binding not in s: s=s.replace(anchor,anchor+binding)
files=sources_to_patch(r)
for rel in files:
    bind='  -v "$ROOT/source/r8-hf2/files/'+rel+':/opt/r8-venv/lib/python3.12/site-packages/'+rel+':ro" '+chr(92)+'\n'
    if bind not in s: s=s.replace(anchor,anchor+bind)
updates={cp:json.dumps(c,indent=2)+'\n',sp:s}
updates.update({r/'source/r8-hf2/files'/rel:data.decode() for rel,data in files.items()})
updates[r/'source/r8-hf2/manifest.json']=json.dumps(dict(update='R8-HF2-Indexer-20261113',
    upstream_head='d423120cdc36d4cf8456f43ddfcc654b0c69c5ea',
    files=[dict(path=rel,before_sha256=CHANGES[rel]['before_sha256'],sha256=hashlib.sha256(data).hexdigest()) for rel,data in files.items()]),indent=2)+'\n' 
backup=r/'runtime'/('hf2-backup-'+str(time.time_ns())); backup.mkdir(mode=0o700)
def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.hf2-',dir=p.parent)
    try:
        os.fchmod(fd,stat.S_IMODE(p.stat().st_mode) if p.exists() else 0o644)
        with os.fdopen(fd,'w') as f: f.write(data)
        os.replace(tmp,p)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
for p,data in updates.items():
    if p.exists():
        dest=backup/p.relative_to(r); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
    write(p,data)
checks=r/'SHA256SUMS'
if checks.exists():
    shutil.copy2(checks,backup/'SHA256SUMS')
    digests={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in updates}
    rows=[]
    for line in checks.read_text().splitlines():
        _,sep,name=line.partition('  ')
        key=name.removeprefix('./')
        rows.append(digests.pop(key)+'  '+name if sep and key in digests else line)
    rows.extend(v+'  '+k for k,v in digests.items()); write(checks,'\n'.join(rows)+'\n')
(r/'runtime/r8-hf2.json').write_text(json.dumps(dict(update='R8-HF2-Indexer-20261113',
    use_local_argmax_reduction=False,dspark_k=5,indexer_metadata='SM80-specific dispatch',image_unchanged=True,
    backup=str(backup),ready='NOT_YET_CONFIRMED'),indent=2)+'\n')
print('R8 HF2 applied. DSpark k5 retained. Backup: '+str(backup))
PY
if [[ "$MODE" == --restart ]]; then
  remove_owned "$FRONTEND"
  remove_owned "$ENGINE"
fi
flock -u 8
exec 8>&-
flock -u 7
exec 7>&-
if [[ "$MODE" == --restart ]]; then
  MODEL="$ROOT/models/DeepSeek-V4.1-Flash-1017"
  [[ ! -f "$STATE/model.path" ]] || IFS= read -r MODEL < "$STATE/model.path"
  LOG="$STATE/start-r8-hf2.log"
  nohup bash "$ROOT/start.sh" --model-dir "$MODEL" --port 8005 > "$LOG" 2>&1 < /dev/null &
  echo "R8 reload started, PID=$!; no device/operator/performance tests."
  echo "Log: $LOG"
  echo "Wait for: Ready: http://127.0.0.1:8005/v1"
fi
