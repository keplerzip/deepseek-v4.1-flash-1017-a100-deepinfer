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
