#!/usr/bin/env python3
"""CPU checks; set DEEPINFER_R2_SOURCE for source-preservation checks.

With the bundled R2 Python, TRITON_INTERPRET=1 additionally executes the
actual candidate Triton kernel on CPU tensors. This is not a GPU gate.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("selected_apply", ROOT / "apply.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)
MANIFEST = json.loads((ROOT / "manifest.json").read_text())
R2_SOURCE = Path(os.environ["DEEPINFER_R2_SOURCE"]) if os.environ.get("DEEPINFER_R2_SOURCE") else None


def split_function():
    source = (ROOT / "fragments/mhc_split.py").read_text()
    namespace = {}
    import builtins
    original_import = builtins.__import__

    def supplied_import(name, *args, **kwargs):
        if name == "vllm.model_executor.kernels.mhc.tilelang":
            return SimpleNamespace(_SMALL_FMA_CONFIG=(6, 16, 128))
        return original_import(name, *args, **kwargs)

    namespace["__builtins__"] = dict(vars(builtins), __import__=supplied_import)
    exec(compile(source, str(ROOT / "fragments/mhc_split.py"), "exec"), namespace)
    return namespace["mhc_fused_post_pre_split_config"]


class SelectedOverlayTests(unittest.TestCase):
    def test_manifest_defaults_and_hashes(self):
        self.assertFalse(MANIFEST["default_enabled"])
        self.assertFalse(MANIFEST["gpu_validated"])
        self.assertTrue(all(value == "0" for value in MANIFEST["flags"].values()))
        for entry in MANIFEST["files"]:
            self.assertEqual(installer.digest((ROOT / entry["fragment"]).read_bytes()), entry["fragment_sha256"])
            self.assertEqual(installer.digest((ROOT / entry["review_patch"]).read_bytes()), entry["patch_sha256"])

    def test_mhc_default_and_explicit_flag(self):
        choose = split_function()
        with patch.dict(os.environ, {"VLLM_DSV41_DEEPINFER_MHC_ALIGNED": "0"}):
            for tokens in range(1, 17):
                self.assertEqual(choose(tokens, 5120, 4), (2 if tokens < 8 else 3, 4, 256))
        with patch.dict(os.environ, {"VLLM_DSV41_DEEPINFER_MHC_ALIGNED": "1"}):
            for tokens in range(1, 17):
                self.assertEqual(choose(tokens, 5120, 4), (6, 8, 128))
            for tokens in (0, 17, 32, 192):
                self.assertIsNone(choose(tokens, 5120, 4))

    def test_mhc_valid_geometry_only(self):
        choose = split_function()
        for enabled in ("0", "1"):
            with patch.dict(os.environ, {"VLLM_DSV41_DEEPINFER_MHC_ALIGNED": enabled}):
                for hidden in (1024, 1536, 2048, 4096, 5120, 7168, 7680, 8192):
                    for hc in (1, 2, 3, 4, 8):
                        for tokens in (1, 6, 8, 12, 16):
                            result = choose(tokens, hidden, hc)
                            if result is not None:
                                tile, splits, threads = result
                                self.assertEqual(hc * (hc + 2) % tile, 0)
                                self.assertEqual(hidden % (splits * threads), 0)

    @unittest.skipUnless(R2_SOURCE, "Provide the exact R2 source tree")
    def test_reconstruction_preserves_unselected_functions(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = installer.prepare(R2_SOURCE, Path(temporary) / "overlay")
            self.assertFalse(report["gpu_validated"])
            for entry in MANIFEST["files"]:
                base = (R2_SOURCE / entry["path"]).read_text()
                derived = (Path(temporary) / "overlay" / entry["path"]).read_text()
                replaced = {item["original"] for item in entry["replacements"]}
                for node in ast.parse(base).body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name not in replaced:
                        self.assertEqual(installer.function_block(base, node.name), installer.function_block(derived, node.name))
                # All class bodies remain byte-identical, including attention,
                # Engram, graph-buffer and ragged metadata paths.
                for node in ast.parse(base).body:
                    if isinstance(node, ast.ClassDef):
                        original = ast.get_source_segment(base, node)
                        self.assertIn(original, derived)
                if "rocm.py" in entry["path"]:
                    reference = installer.function_block(derived, "_deepinfer_combine_topk_swa_indices_torch_reference")
                    self.assertEqual(reference.replace("def _deepinfer_combine_topk_swa_indices_torch_reference(", "def combine_topk_swa_indices("), installer.function_block(base, "combine_topk_swa_indices"))

    @unittest.skipUnless(R2_SOURCE, "Provide the exact R2 source tree")
    def test_modified_source_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            target = source / MANIFEST["files"][0]["path"]
            target.parent.mkdir(parents=True)
            target.write_text((R2_SOURCE / MANIFEST["files"][0]["path"]).read_text() + "\n# changed\n")
            output = Path(temporary) / "output"
            with self.assertRaisesRegex(ValueError, "pinned R2"):
                installer.prepare(source, output)
            self.assertFalse(output.exists())

    @unittest.skipUnless(R2_SOURCE, "Provide the exact R2 source tree")
    def test_in_place_output_rejected(self):
        with self.assertRaisesRegex(ValueError, "separate"):
            installer.prepare(R2_SOURCE, R2_SOURCE)
        with self.assertRaisesRegex(ValueError, "separate"):
            installer.prepare(R2_SOURCE, R2_SOURCE / "unsafe-output")

    @unittest.skipUnless(os.environ.get("TRITON_INTERPRET") == "1" and R2_SOURCE,
                         "Requires R2 torch/Triton CPU interpreter")
    def test_actual_triton_cpu_interpreter_against_r2_torch(self):
        import torch
        import triton
        import triton.language as tl
        platform = SimpleNamespace(is_cuda=lambda: True,
                                   get_device_capability=lambda: SimpleNamespace(major=8))
        namespace = runpy.run_path(str(ROOT / "fragments/rocm_combine.py"), init_globals={
            "torch": torch, "triton": triton, "tl": tl, "os": os,
            "current_platform": platform, "_SPARSE_PREFILL_TOPK_ALIGNMENT": 128,
        })
        base = (R2_SOURCE / MANIFEST["files"][0]["path"]).read_text()
        reference_source = installer.function_block(base, "combine_topk_swa_indices")
        reference_namespace = {"torch": torch, "_SPARSE_PREFILL_TOPK_ALIGNMENT": 128}
        exec(compile(reference_source, "r2-torch-reference", "exec"), reference_namespace)
        reference = reference_namespace["combine_topk_swa_indices"]
        namespace["combine_topk_swa_indices"].__globals__["_deepinfer_combine_topk_swa_indices_torch_reference"] = reference
        cases = [([1], [1], 0, 1, 16, 128), ([6], [73], 8, 4, 32, 128),
                 ([3, 0, 5], [120, 64, 320], 17, 4, 16, 128),
                 ([2, 7], [2, 19], 100, 1, 32, 8),
                 ([16], [2048], 0, 4, 48, 128),
                 ([17, 4, 2], [1024, 300, 700], 113, 4, 16, 128),
                 ([6], [3], 29, 4, 32, 128),
                 ([128], [64000], 77, 4, 384, 128),
                 ([12], [512], 9, 1, 16, 13)]
        generator = torch.Generator().manual_seed(1017)
        for lengths, sequences, base_index, ratio, width, window in cases:
            rows = sum(lengths)
            values = torch.randint(-3, 1027, (rows, width), dtype=torch.int32, generator=generator)
            starts = [base_index]
            for length in lengths:
                starts.append(starts[-1] + length)
            query = torch.tensor(starts, dtype=torch.int32)
            seq = torch.tensor(sequences, dtype=torch.int32)
            gather = torch.minimum(seq, torch.full_like(seq, 256))
            arguments = (values, query, seq, gather, window, ratio, width + 4, 4096, 1024)
            expected = reference(*arguments)
            with patch.dict(os.environ, {"VLLM_DSV41_DEEPINFER_FUSED_COMBINE": "1"}):
                actual = namespace["combine_topk_swa_indices"](*arguments)
            for candidate, baseline in zip(actual, expected):
                self.assertTrue(torch.equal(candidate, baseline), str((lengths, ratio, width, window)))
            with patch.dict(os.environ, {"VLLM_DSV41_DEEPINFER_FUSED_COMBINE": "0"}):
                baseline_mode = namespace["combine_topk_swa_indices"](*arguments)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(baseline_mode, expected)))
        with patch.dict(os.environ, {"VLLM_DSV41_DEEPINFER_FUSED_COMBINE": "1"}):
            empty = namespace["combine_topk_swa_indices"](
                torch.empty((0, 32), dtype=torch.int32),
                torch.tensor([17], dtype=torch.int32),
                torch.empty((0,), dtype=torch.int32), torch.empty((0,), dtype=torch.int32),
                128, 4, 32, 4096, 1024,
            )
            self.assertEqual(empty[0].shape, (0, 256))
            self.assertEqual(empty[1].shape, (0,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
