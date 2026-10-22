# Selected SM80 experiment

This directory contains two function-level adaptations from
[HalfVulpes/Deepinfer-SM80](https://github.com/HalfVulpes/Deepinfer-SM80), pinned at
`2a828fc2122b6fc59805c0c98b6aaf78e7da9b48`. Both are disabled by default. They
affect GPU computation; they do not implement or change tool calling.

| Candidate | Selected change | Preserved implementation |
|---|---|---|
| `topk_swa` | Fuse prefill TopK + sliding-window index assembly into a Triton kernel with disjoint store ranges | Keep the original Torch function as an executable reference; retain all attention classes, fused inverse RoPE, ragged reuse and graph buffers |
| `mhc_aligned` | Add an aligned `(tile_n=6, splits=8, threads=128)` choice for small-token shapes such as `hidden_size=5120` | Keep R2's existing fused post/pre implementation, auxiliary-state capture ABI and warmup registration |

The original community patch does not apply to four of its six target files in
this project's exact R2 image. Copying all six modules would also import a
missing `pp_share` module and newer Engram methods. The full sparse-attention
replacement removes this project's fused inverse RoPE, blocked prefill and
8192-row fast-scan changes. Those files are not adopted here. The UE8M0/LUT
candidate is deferred pending edge-exponent and GPU numerical checks.

`fragments/` contains only selected functions. `apply.py` reconstructs two
complete overlay modules from their exact R2 SHA256 baselines. All other
functions and class bodies remain unchanged. The original source directory is
never modified. `patches/` records the exact, reviewable function edits.

Inside the existing R2 runtime, prepare the overlay into a separate directory:

```bash
python3 source/sm80-overlay/apply.py \
  --source-root /usr/local/lib/python3.12/dist-packages \
  --output-root /tmp/selected-sm80 \
  --report /tmp/selected-sm80-source-check.json
```

`--check-only` validates reconstruction without creating an overlay. The
manifest binds the image, baseline modules, fragments, review patches and
generated output hashes. Different source bytes fail before output is written.

Run the CPU tests using the bundled R2 Python. A read-only container needs an
ephemeral writable `/tmp` for Torch and the test scratch files:

```bash
TRITON_INTERPRET=1 \
DEEPINFER_R2_SOURCE=/usr/local/lib/python3.12/dist-packages \
python3 source/sm80-overlay/tests/test_selected_overlay.py
```

These checks execute the actual Triton combine kernel through its CPU
interpreter and compare int32 indices/lengths with the preserved R2 Torch
reference. They also verify flag-off behavior, mHC shape selection, rejection
of changed baseline files and preservation of unrelated source. CPU interpreter
execution does not validate CUDA races, SM80 numerical kernels, graph replay,
eight-GPU inference or performance.

GPU acceptance must use the exact generated hashes and the candidate flags:

```text
VLLM_DSV41_DEEPINFER_FUSED_COMBINE=1
VLLM_DSV41_DEEPINFER_MHC_ALIGNED=1
```

The ordinary runtime uses `0` for both. A hash-bound GPU acceptance report and
an explicit candidate launch are required before using either flag in a
service. No performance gain has been measured on the build host.
