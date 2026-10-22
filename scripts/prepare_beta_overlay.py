"""Prepare SHA-bound experimental source and review diffs from exact R2 files."""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(sys.argv[1])
OUT = ROOT / "source/beta-overlay"
UP = ROOT / "vendor/sm80/upstream/Deepinfer-SM80/vllm_sm80_deepinfer"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def block(source, name):
    node, = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
    return "\n".join(source.splitlines()[start:node.end_lineno])


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one source fragment: {old[:100]}")
    return source.replace(old, new, 1)


files = []


def save(path, source):
    original = (BASE / path).read_bytes()
    compile(source, path, "exec")
    dest = OUT / "files" / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(source)
    ref = ROOT / "vendor/r2-beta-source" / path
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_bytes(original)
    patch = OUT / "patches" / (path.replace("/", "__") + ".patch")
    patch.parent.mkdir(parents=True, exist_ok=True)
    patch.write_text("".join(difflib.unified_diff(original.decode().splitlines(True), source.splitlines(True),
                                               fromfile="r2/"+path, tofile="beta/"+path)))
    files.append({"path": path, "base_sha256": sha(original), "beta_sha256": sha(source.encode()),
                  "patch": str(patch.relative_to(OUT)), "patch_sha256": sha(patch.read_bytes())})


path = "vllm/models/deepseek_v4_1/common/ops/cache_utils.py"
s = (BASE / path).read_text()
s = replace_once(s, "        scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)",
                 "        bits = encoded_scale.to(tl.int32) << 23\n"
                 "        bits = tl.where(encoded_scale == 0, 1 << 22, bits)\n"
                 "        scale = bits.to(tl.float32, bitcast=True)")
save(path, s)

path = "vllm/v1/attention/ops/rocm_aiter_mla_sparse.py"
s = (BASE / path).read_text()
s = s.replace("import functools\n", "import functools\nimport os\n", 1)
up = (UP / path).read_text()
helper = block(up, "_decode_e8m0_scales_bf16_triton").replace("tl.where(encoded_scales == 0, 0, bits)", "tl.where(encoded_scales == 0, 64, bits)")
partial = block(up, "_sparse_attn_decode_partial_kernel")
old = block(s, "_sparse_attn_decode_partial_kernel")
# Keep an independent baseline for numerical/performance A/B in the same image.
ref = old.replace("def _sparse_attn_decode_partial_kernel(", "def _beta_reference_decode_partial_kernel(", 1)
s = replace_once(s, old, ref + "\n\n" + helper + "\n\n" + partial)
s = s.replace("tl.exp2(encoded_scales.to(tl.float32) - 127.0)", "_decode_e8m0_scales_triton(encoded_scales)")
# Dispatch preserves R2 fused inverse RoPE and its reduction ABI.
s = replace_once(s, "        _sparse_attn_decode_partial_kernel[(num_queries, num_splits, heads_blocks)](",
    "        beta_unified = os.environ.get('DEEPINFER_BETA_UNIFIED_MLA', '1') == '1'\n"
    "        partial_kernel = (_sparse_attn_decode_partial_kernel if beta_unified\n"
    "                          else _beta_reference_decode_partial_kernel)\n"
    "        partial_kernel[(num_queries, num_splits, heads_blocks)](")
s = replace_once(s, "            num_warps=8 if current_platform.is_cuda() else 4,",
    "            num_warps=(4 if beta_unified and base_workgroups >= 12 else 8)\n"
    "                      if current_platform.is_cuda() else 4,")
save(path, s)

path = "vllm/models/deepseek_v4_1/common/engram.py"
s = (BASE / path).read_text()
old = '''    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        """Gather this layer's rows on the main stream before decoder layers."""
        self.embed_tokens.lookup(hash_ids, self.staged_rows[: hash_ids.shape[0]])'''
new = '''    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        """Side-stream lookup; join only when this layer consumes its rows."""
        if not self.beta_prefetch:
            self.embed_tokens.lookup(hash_ids, self.staged_rows[: hash_ids.shape[0]])
            return
        current = torch.cuda.current_stream()
        self.beta_stream.wait_stream(current)
        with torch.cuda.stream(self.beta_stream):
            self.embed_tokens.lookup(hash_ids, self.staged_rows[: hash_ids.shape[0]], background=True)
            self.beta_ready.record(self.beta_stream)'''
s = replace_once(s, old, new)
s = replace_once(s, "        self.layer_hash_index = layer_hash_index",
    "        import os\n"
    "        self.beta_prefetch = os.environ.get('DEEPINFER_BETA_ENGRAM_PREFETCH', '1') == '1'\n"
    "        self.beta_stream = torch.cuda.Stream() if self.beta_prefetch else None\n"
    "        self.beta_ready = torch.cuda.Event() if self.beta_prefetch else None\n"
    "        self.layer_hash_index = layer_hash_index")
s = replace_once(s, '        rows = self.staged_rows[: hash_ids.shape[0]]',
    '        if self.beta_prefetch:\n'
    '            torch.cuda.current_stream().wait_event(self.beta_ready)\n'
    '        rows = self.staged_rows[: hash_ids.shape[0]]')
save(path, s)

path = "vllm/models/deepseek_v4_1/nvidia/model.py"
s = (BASE / path).read_text()
s = replace_once(s, "        self.hidden_size = config.hidden_size",
    "        self.beta_layer_id = extract_layer_index(prefix)\n"
    "        self.hidden_size = config.hidden_size")
s = replace_once(s, "        x = self.ffn(x, input_ids)",
    "        from vllm.models.deepseek_v4_1.beta_runtime import consumer_ffn\n"
    "        x = consumer_ffn(self, x, input_ids, positions)")
save(path, s)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "manifest.json").write_text(json.dumps({
    "release": "20261022-r3-cpukv", "gpu_validated": False, "default_enabled": True,
    "yoco_ffn_default_enabled": False,
    "approximate": ["YOCO consumer FFN compaction outside per-request/checkpoint tails"],
    "files": files,
}, indent=2) + "\n")
print(json.dumps({"files": len(files), "status": "PREPARED", "gpu": "NOT_RUN"}))
