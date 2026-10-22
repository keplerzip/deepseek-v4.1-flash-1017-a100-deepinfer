# HF1–HF5

- HF1: use the active GPU device's CUDA stream for each graph capture in the eight-device kernel gates.
- HF2: restrict native CPU offloading to prefix-cacheable groups, preserve original group mappings and allocation protection, and exclude circular scratch groups. This code is optional in the successful deployment.
- HF3: experimental contiguous per-rank CPU backing and large host-registration batches for a 768 GiB total CPU KV target, plus mapping-local NUMA interleave. The deployment owner abandoned CPU KV offload; it is not the performance baseline.
- HF4: default CPU KV pool to zero; set GPU memory utilization to 0.95; invalidate the cached middleware stack only during pre-serving app construction so the pinned native bootstrap can accept the official routes and trace middleware. Native middleware remains registered.
- HF5: decode synthetic benchmark filler with the checkpoint's tokenizers decoder because recipe.Tokenizer provides encode, not decode; validate re-encoded filler size and actual input usage. Future launches bind the corrected script. All original target acceptance gates remain enabled.

The current snapshot includes these fixes. Protocol conversion, tool parsing and upstream server source retain official recipe behavior. Approximate YOCO FFN is off. Historical research documents may mention the original 0.92 GPU / 768 GiB CPU KV settings; current defaults and reported measurements are documented in the root README and configs/runtime.json.
