#!/usr/bin/env python3
"""Compile selected combine specializations for SM80 without a GPU device."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy


def main():
    import torch
    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "manifest.json").read_text())
    namespace = runpy.run_path(str(root / "fragments/rocm_combine.py"), init_globals={
        "torch": torch, "triton": triton, "tl": tl, "os": os,
    })
    function = namespace["_deepinfer_combine_topk_swa_indices_kernel"]
    rows = []
    for topk, ratio, window in ((384, 4, 128), (512, 1, 128), (16, 4, 128), (0, 1, 128)):
        constants = {"TOP_K": topk, "COMPRESS_RATIO": ratio, "WINDOW_SIZE": window,
                     "TOPK_WIDTH": max(1, topk),
                     "PADDED_TOP_K": max(16, triton.next_power_of_2(max(1, topk))),
                     "COMBINED_TOPK": triton.cdiv(topk + window, 128) * 128}
        signature = {name: ("constexpr" if name in constants else "*i32" if name.endswith("_ptr") else "i32")
                     for name in function.arg_names}
        kernel = triton.compile(ASTSource(function, signature, constants),
                                target=GPUTarget("cuda", 80, 32), options={"num_warps": 4})
        rows.append({"constants": constants, "shared_memory_bytes": kernel.metadata.shared,
                     "artifact_formats": list(kernel.asm), "status": "PASS"})
    report = {"status": "PASS", "scope": "SM80 offline PTX/cubin compilation only",
              "base_image_id": manifest["base_image_id"], "kernel": function.__name__,
              "fragment_sha256": hashlib.sha256((root / "fragments/rocm_combine.py").read_bytes()).hexdigest(),
              "gpu_executed": False, "gpu_validated": False, "cases": rows}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
