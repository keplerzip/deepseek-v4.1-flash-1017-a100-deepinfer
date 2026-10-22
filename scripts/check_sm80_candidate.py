#!/usr/bin/env python3
"""Fail-closed static audit and per-device numerical gate for selected SM80 kernels.

Static PASS never enables candidates. GPU PASS is scoped to these two kernels;
it does not certify whole-model quality, speed, C32, or 512K serving.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "source/sm80-overlay/manifest.json"
FLAGS = ("VLLM_DSV41_DEEPINFER_FUSED_COMBINE", "VLLM_DSV41_DEEPINFER_MHC_ALIGNED")
UPSTREAM_COMMIT = "2a828fc2122b6fc59805c0c98b6aaf78e7da9b48"
REQUIRED_CANDIDATES = {"topk_swa", "mhc_aligned"}


class NotRun(RuntimeError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_manifest(path=MANIFEST):
    manifest = json.loads(Path(path).read_text())
    require(manifest["source"]["commit"] == UPSTREAM_COMMIT, "Unexpected upstream source commit")
    require(manifest["default_enabled"] is False, "Candidates must remain disabled by default")
    require(all(manifest["flags"].get(f) == "0" for f in FLAGS), "Unsafe default feature flags")
    require(len(manifest["files"]) == 2, "Expected precisely two selected target files")
    return manifest


def static_check(path=MANIFEST):
    manifest = read_manifest(path)
    directory = Path(path).parent
    records = []
    for entry in manifest["files"]:
        fragment = directory / entry["fragment"]
        require(sha256(fragment) == entry["fragment_sha256"], f"Fragment hash mismatch: {fragment}")
        ast.parse(fragment.read_text(), filename=str(fragment))
        require((directory / entry["review_patch"]).is_file(), "Missing review patch")
        target = Path(entry["path"])
        require(not target.is_absolute() and ".." not in target.parts, "Unsafe package target")
        records.append({"path": entry["path"], "fragment_sha256": sha256(fragment), "syntax": "PASS"})
    provenance_path = ROOT / "vendor/sm80/provenance.json"
    provenance = json.loads(provenance_path.read_text())
    require(provenance["commit"] == UPSTREAM_COMMIT, "Vendor provenance commit mismatch")
    for entry in provenance["files"]:
        target = ROOT / "vendor/sm80" / entry["path"]
        require(sha256(target) == entry["sha256"], f"Vendored source hash mismatch: {target}")
        if target.suffix == ".py":
            ast.parse(target.read_text(), filename=str(target))
    return manifest, records


def installed_hashes(manifest, package_root):
    root = Path(package_root).resolve(strict=True)
    result = {}
    for entry in manifest["files"]:
        target = (root / entry["path"]).resolve(strict=True)
        require(target.is_relative_to(root), "Package file escapes package root")
        result[entry["path"]] = sha256(target)
        require(result[entry["path"]] == entry["patched_sha256"], f"Installed hash mismatch: {target}")
        ast.parse(target.read_text(), filename=str(target))
    return result


def validate_image_id(value):
    require(isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value), "A full immutable sha256 image ID is required")
    return value


def gpu_inventory(torch, expected_gpus):
    if not torch.cuda.is_available():
        raise NotRun("CUDA device unavailable; no numerical or graph checks ran")
    count = torch.cuda.device_count()
    require(count == expected_gpus, f"Expected {expected_gpus} visible GPUs, got {count}")
    devices = []
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        require((properties.major, properties.minor) == (8, 0), f"GPU {index} is not SM80")
        uuid = str(getattr(properties, "uuid", ""))
        require(uuid and uuid not in ("None", ""), "PyTorch must expose CUDA GPU UUID for evidence binding")
        devices.append({"index": index, "uuid": uuid, "name": properties.name,
                        "capability": [properties.major, properties.minor],
                        "sm_count": properties.multi_processor_count, "total_memory": properties.total_memory})
    require(len({d["uuid"] for d in devices}) == count, "Duplicate CUDA GPU UUIDs")
    return devices


def evidence_binding(report, manifest, hashes, image_id, devices):
    require(report.get("schema_version") == 1 and report.get("kind") == "sm80_candidate_gate", "Wrong evidence schema")
    require(report.get("status") == "PASS" and report.get("mode") == "gpu", "Only a completed GPU PASS is usable")
    require(report.get("manifest_sha256") == sha256(MANIFEST), "Stale manifest evidence")
    require(report.get("base_image_id") == manifest["base_image_id"], "Base image mismatch")
    require(report.get("image_id") == image_id, "Runtime image mismatch")
    require(report.get("installed_files") == hashes, "Installed files do not match GPU evidence")
    require(set(report.get("validated_candidates", [])) == REQUIRED_CANDIDATES, "Candidate evidence is incomplete")
    require(report.get("gpu_devices") == devices, "GPU inventory differs from validated inventory")
    require(bool(report.get("checks", {}).get("gpu")), "Missing numerical checks")


def reference_combine(rows, query_lens, seq_lens, gather_lens, window, ratio, topk, m, n):
    """Independent scalar CPU oracle; preserves every padding cell and invalid index."""
    width = ((topk + window + 127) // 128) * 128
    result, lengths = [], []
    row_id = 0
    for request, query_len in enumerate(query_lens):
        for token in range(query_len):
            position = seq_lens[request] - query_len + token
            compressed = min(max((position + 1) // ratio, 0), topk, len(rows[row_id]))
            swa = max(min(position + 1, window), 0)
            values = [-1] * width
            for column in range(compressed):
                value = rows[row_id][column]
                values[column] = value + m * request if 0 <= value < n else -1
            for column in range(swa):
                values[compressed + column] = m * request + n + column + position - swa + 1 - (seq_lens[request] - gather_lens[request])
            result.append(values)
            lengths.append(compressed + swa)
            row_id += 1
    return result, lengths


def import_checked(module_name, package_root, manifest):
    module = importlib.import_module(module_name)
    path = Path(module.__file__).resolve()
    expected = Path(package_root).resolve() / (module_name.replace(".", "/") + ".py")
    require(path == expected.resolve(), f"Imported a different installation: {path}")
    matching = [f for f in manifest["files"] if f["path"] == module_name.replace(".", "/") + ".py"]
    require(len(matching) == 1 and sha256(path) == matching[0]["patched_sha256"], "Imported module hash mismatch")
    return module


def test_topk(torch, module, device):
    shapes = [([1], 0, 512, 2), ([7], 7, 512, 2), ([32], 512, 512, 1),
              ([127], 512, 512, 2), ([129], 512, 512, 4), ([1536], 512, 512, 2),
              ([8192], 512, 512, 2), ([1, 0, 7, 33], 17, 512, 2),
              ([2, 3], 1024, 1024, 1)]
    generator = torch.Generator().manual_seed(20261021)
    checks = []
    for number, (query_lens, row_width, requested, ratio) in enumerate(shapes):
        count = sum(query_lens)
        n = max(2048, count * 2)
        lengths = [q + (0 if number % 2 else 131072) for q in query_lens]
        gathers = [min(length, 128 + q) for length, q in zip(lengths, query_lens)]
        m = n + max(gathers) + 256
        rows_cpu = torch.randint(-3, n + 4, (count, row_width), generator=generator, dtype=torch.int32)
        starts = [73]
        for q in query_lens:
            starts.append(starts[-1] + q)
        args = (rows_cpu.to(device), torch.tensor(starts, device=device, dtype=torch.int32),
                torch.tensor(lengths, device=device, dtype=torch.int32),
                torch.tensor(gathers, device=device, dtype=torch.int32), 128, ratio, requested, m, n)
        oracle_rows, oracle_lens = reference_combine(rows_cpu.tolist(), query_lens, lengths, gathers, 128, ratio, requested, m, n)
        expected = (torch.tensor(oracle_rows, device=device, dtype=torch.int32), torch.tensor(oracle_lens, device=device, dtype=torch.int32))
        os.environ[FLAGS[0]] = "1"
        actual = module.combine_topk_swa_indices(*args)
        torch.cuda.synchronize(device)
        for got, want in zip(actual, expected):
            require(got.dtype == torch.int32 and got.shape == want.shape, "TopK shape or dtype mismatch")
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        os.environ[FLAGS[0]] = "0"
        baseline = module.combine_topk_swa_indices(*args)
        for got, want in zip(baseline, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        replayed = number in (2, 7)
        if replayed:
            os.environ[FLAGS[0]] = "1"
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize(device)
            with torch.cuda.graph(graph, stream=torch.cuda.Stream(device=device)):
                captured = module.combine_topk_swa_indices(*args)
            graph.replay()
            graph.replay()
            torch.cuda.synchronize(device)
            for got, want in zip(captured, expected):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
        checks.append({"candidate": "topk_swa", "device": str(device), "query_lens": query_lens,
                       "input_width": row_width, "topk": requested, "compress_ratio": ratio,
                       "base_offset": 73, "oracle": "scalar_cpu_and_pinned_torch", "status": "PASS",
                       "exact": True, "cuda_graph_replayed": replayed})
    # Empty input is a separate shape contract: the pinned Torch baseline's
    # reshape(0, -1) cannot express it, so validate against the explicit contract.
    os.environ[FLAGS[0]] = "1"
    empty = module.combine_topk_swa_indices(torch.empty((0, 512), device=device, dtype=torch.int32),
             torch.tensor([0], device=device, dtype=torch.int32), torch.empty(0, device=device, dtype=torch.int32),
             torch.empty(0, device=device, dtype=torch.int32), 128, 2, 512, 4096, 2048)
    require(empty[0].shape == (0, 640) and empty[1].shape == (0,), "Empty TopK contract mismatch")
    checks.append({"candidate": "topk_swa", "device": str(device), "shape": "empty", "status": "PASS"})
    return checks


def test_mhc(torch, module, device):
    checks = []
    generator = torch.Generator(device=device).manual_seed(20261021)
    def random(shape, dtype=torch.float32):
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)
    hidden, hc = 5120, 4
    for tokens in (1, 6, 12, 16, 17, 32):
        for with_options in (False, True):
            x = random((tokens, hidden), torch.bfloat16)
            residual = random((tokens, hc, hidden), torch.bfloat16)
            post = torch.sigmoid(random((tokens, hc, 1)))
            comb = torch.softmax(random((tokens, hc, hc)), -1)
            fn = random((hc * (hc + 2), hc * hidden)) * 0.01
            scale = torch.tensor([0.1, 0.1, 0.1], device=device)
            base = torch.zeros(hc * (hc + 2), device=device)
            pre = torch.sigmoid(random((tokens, hc))) if with_options else None
            norm = torch.ones(hidden, device=device, dtype=torch.bfloat16) if with_options else None
            args = (x, residual, post, comb, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20)
            kwargs = {"pre_mix": pre, "norm_weight": norm, "capture_aux": with_options}
            os.environ[FLAGS[1]] = "0"
            baseline = module.mhc_fused_post_pre_delayed_tilelang(*args, **kwargs)
            base_geometry = module.mhc_fused_post_pre_split_config(tokens, hidden, hc)
            os.environ[FLAGS[1]] = "1"
            geometry = module.mhc_fused_post_pre_split_config(tokens, hidden, hc)
            require(geometry == ((6, 8, 128) if tokens <= 16 else None), "Aligned geometry was not selected")
            if tokens <= 16:
                require(base_geometry != geometry, "Candidate did not exercise a changed configuration")
            actual = module.mhc_fused_post_pre_delayed_tilelang(*args, **kwargs)
            require(len(actual) == len(baseline) == 6, "R2 capture_aux ABI changed")
            errors = []
            for index, (got, want) in enumerate(zip(actual, baseline)):
                require(got.shape == want.shape and got.dtype == want.dtype, f"mHC output {index} shape/dtype mismatch")
                require(bool(torch.isfinite(got).all()), f"Nonfinite mHC output {index}")
                # Residual and capture_aux depend only on the unchanged post map.
                if index in (0, 5) or tokens > 16:
                    torch.testing.assert_close(got, want, rtol=0, atol=0)
                else:
                    rtol, atol = ((5e-3, 2e-3) if got.dtype == torch.bfloat16 else (2e-4, 2e-5))
                    torch.testing.assert_close(got, want, rtol=rtol, atol=atol)
                errors.append(float((got.float() - want.float()).abs().max()) if got.numel() else 0.0)
            if tokens in (6, 12, 16) and with_options:
                graph = torch.cuda.CUDAGraph()
                # Every specialization was warmed above. Only graph replay is
                # accepted here; JIT compilation success alone is insufficient.
                torch.cuda.synchronize(device)
                with torch.cuda.graph(graph, stream=torch.cuda.Stream(device=device)):
                    captured = module.mhc_fused_post_pre_delayed_tilelang(*args, **kwargs)
                graph.replay()
                graph.replay()
                torch.cuda.synchronize(device)
                for got, want in zip(captured, actual):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)
            checks.append({"candidate": "mhc_aligned", "device": str(device), "tokens": tokens,
                           "hidden_size": hidden, "hc_mult": hc, "norm_pre_aux": with_options,
                           "geometry": geometry, "max_abs_errors": errors,
                           "cuda_graph_replayed": tokens in (6, 12, 16) and with_options, "status": "PASS"})
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "gpu", "verify"), default="static")
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--image-id")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-gpus", type=int, default=8)
    args = parser.parse_args(argv)
    require(args.expected_gpus > 0, "expected-gpus must be positive")
    if args.mode == "verify":
        require(args.report.is_file(), "GPU evidence file does not exist")
        evidence = json.loads(args.report.read_text())
    else:
        evidence = None
    report = {"schema_version": 1, "kind": "sm80_candidate_gate", "mode": args.mode,
              "status": "NOT_RUN", "created_at": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": sha256(MANIFEST), "validated_candidates": [], "checks": {},
              "scope": "Selected kernel numerics and CUDA graph compatibility only; no model or serving acceptance"}
    exit_code = 1
    saved_flags = {flag: os.environ.get(flag) for flag in FLAGS}
    try:
        manifest, records = static_check()
        report.update(base_image_id=manifest["base_image_id"], checks={"static": records})
        if args.mode == "static":
            report.update(status="PASS", gpu_status="NOT_RUN")
        else:
            require(args.package_root is not None, "package-root is required")
            validate_image_id(args.image_id)
            hashes = installed_hashes(manifest, args.package_root)
            report.update(image_id=args.image_id, installed_files=hashes)
            try:
                import torch
            except ImportError as error:
                raise NotRun("PyTorch is unavailable; run GPU gate inside the candidate image") from error
            devices = gpu_inventory(torch, args.expected_gpus)
            report["gpu_devices"] = devices
            if args.mode == "verify":
                evidence_binding(evidence, manifest, hashes, args.image_id, devices)
                print(json.dumps({"status": "PASS", "mode": "verify", "validated_candidates": sorted(REQUIRED_CANDIDATES)}))
                return 0
            sys.path.insert(0, str(args.package_root.resolve()))
            os.environ.update({flag: "0" for flag in FLAGS})
            topk = import_checked("vllm.models.deepseek_v4_1.amd.rocm", args.package_root, manifest)
            mhc = import_checked("vllm.model_executor.kernels.mhc.dsv41_sm80", args.package_root, manifest)
            report["runtime_versions"] = {"torch": torch.__version__, "cuda": torch.version.cuda}
            gpu_checks = []
            for info in devices:
                print(json.dumps({"stage":"selected_gpu_case","gpu":info["index"]}),flush=True)
                torch.cuda.set_device(info["index"])
                device = torch.device("cuda", info["index"])
                gpu_checks.extend(test_topk(torch, topk, device))
                gpu_checks.extend(test_mhc(torch, mhc, device))
            report["checks"]["gpu"] = gpu_checks
            report.update(status="PASS", validated_candidates=sorted(REQUIRED_CANDIDATES))
        exit_code = 0
    except NotRun as error:
        report.update(status="NOT_RUN", reason=str(error))
        exit_code = 2
    except Exception as error:
        report.update(status="FAIL", reason=f"{type(error).__name__}: {error}")
    finally:
        for flag, value in saved_flags.items():
            if value is None:
                os.environ.pop(flag, None)
            else:
                os.environ[flag] = value
    # Verification is read-only: never replace a successful GPU report with
    # a failed verification report or a static check.
    if args.mode != "verify":
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(args.report.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.report)
    print(json.dumps({key: report.get(key) for key in ("status", "mode", "reason", "gpu_status")}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
