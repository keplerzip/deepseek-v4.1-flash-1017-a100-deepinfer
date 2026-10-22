# Runtime dependencies and reproduction boundary

This repository is the complete deployment-layer source snapshot for the reported HF5 setup. It does not contain the entire vLLM/CUDA runtime or model weights.

- Runtime base: local image `deepseek-v4.1-flash-1017-a100:20261018-r2`, config digest `sha256:a63405a72ce202bb8347f17fc2432be0df484b794fba0224fd210b6daff004f6`. A public registry distribution/build chain is not provided here.
- Existing formal R3 image identities: `manifests/runtime.env`; original filenames/tag strings retain `cpukv` for compatibility, while the current default CPU KV pool is **0**.
- Base vLLM/backport source: `a350766628514d597670d2a5217777fdee2ba7f4`; the local R2 image is stricter than a generic same-named upstream vLLM version. Arbitrary replacements are unsupported by source hash gates.
- DS official recipe: 0.2.0, Linux x86_64 cp310-abi3 wheel, SHA256 `259812bf382bc82668b672707c5922095cc4f5f60a6c514ff89640bb62a746f7`; wheel and third-party notices are included for offline runtime-layer builds.
- Model metadata/tokenizer: pinned versions and checksums in `vendor/model-manifest.json`; only tokenizer/metadata, no model weights, are distributed.
- Reference kernel provenance: `vendor/sm80/provenance.json`; official recipe provenance: `vendor/official-recipe/provenance.json`.

The tested machine used 8 A100 80 GiB GPUs and roughly 2 TiB host RAM. Minimum RAM was not measured. Engram parameter CPU offload remains enabled; the experimental CPU KV tier is disabled. GPU/driver/kernel versions were not included in the public user-reported performance summary.

For offline packaging from a supported local base: build the runtime layer, export the image, then create the package with the bundled scripts. Public GitHub source-check CI has no model/GPUs and must not be treated as model or performance validation.
