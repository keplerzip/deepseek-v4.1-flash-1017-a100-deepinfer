R8 HF2 cumulative metadata fix / 2026-11-13

Apply as the original deployment user:
  sudo -v
  bash apply.sh /ai/services/deepseek-v4.1-flash-1017-a100-deepinfer-r8/deepseek-v4.1-flash-1017-a100-deepinfer-r8

Includes HF1: disables use_local_argmax_reduction (DSpark has no get_top_tokens).
HF2 adds a virtual cache-format resolver to the stock indexer metadata builder
and overrides it only for AmpereIndexerMetadataBuilder. The SM80 path uses its
existing MXFP4 codecs and BF16 kernels. The stock Blackwell/CDNA hardware guard
is unchanged. No numerical kernels or official DS protocol parsers are edited.

The updater reads and hashes the two exact pinned sources already shipped in
vendor/upstream-pr66497-source.tar.gz, derives the patch and binds the two files
read-only into the R8 engine. No downloads, recompilation or image replacement.
Included files/ and manifest.json let you inspect the exact patched sources.

The updater backs up modified files, validates ownership, stops/removes only
this installation's R8 containers, waits for the previous startup lock to
release, and reloads R8. Missing frontend or engine containers are accepted.
Reapplying is supported; a second --no-restart argument skips the reload.
R7 containers, weights and the model soft link are retained. Configuration:
8005 / .94 GPU utilization / 524288 context / DSpark k5 / official DS protocol.
No device checks, operator checks or performance tests are launched.

Reload log: runtime/start-r8-hf2.log
Success line: Ready: http://127.0.0.1:8005/v1
A "reload started" message alone does not establish readiness.

Local validation: 17 CPU tests PASS. Actual stock/Ampere constructors were run
with CPU tensors and simulated SM80 capability. The old dispatch reproduces
the exact target exception; the fixed constructor completes for compression
ratios 1, 4 and 128. Stock hardware checks, invalid cache/device rejection,
source sealing, config mounts, reapplication, ownership and reload flow pass.
No actual A100 startup, inference, tool quality or 512K model execution was
performed locally. This patch addresses the reported metadata-dispatch error.
