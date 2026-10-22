#!/usr/bin/env python3
"""Read-only official 1017 snapshot verification and fail-closed A100 resources."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
MODEL_REVISION = "f04a978641fa9bfe6f448ec5da3b2e1c2520a6ad"
DEFAULT_MANIFEST = ROOT / "vendor/model-manifest.json"
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E8M0": 1,
               "F8_E5M2": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
               "F32": 4, "I32": 4, "U32": 4, "F64": 8, "I64": 8, "U64": 8}


class NoGpu(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path=DEFAULT_MANIFEST):
    manifest = json.loads(Path(path).read_text())
    require(manifest.get("source") == "ModelScope", "Expected original ModelScope manifest")
    require(manifest.get("modelscope_model_id") == MODEL_ID, "Wrong model identity")
    require(manifest.get("modelscope_revision") == MODEL_REVISION, "Wrong official 1017 revision")
    files = manifest.get("files", [])
    require(len(files) == 84, "Official manifest must contain exactly 84 files")
    names = [entry["path"] for entry in files]
    require(len(set(names)) == 84, "Duplicate manifest file paths")
    require(sum(name.endswith(".safetensors") for name in names) == 48, "Expected 48 weight shards")
    for entry in files:
        name = Path(entry["path"])
        require(not name.is_absolute() and ".." not in name.parts, "Unsafe manifest path")
        require(isinstance(entry["size"], int) and entry["size"] >= 0, "Invalid manifest size")
        require(len(entry["sha256"]) == 64 and all(c in "0123456789abcdef" for c in entry["sha256"]), "Invalid manifest SHA256")
    require(sum(entry["size"] for entry in files) == manifest["snapshot_bytes"], "Manifest byte total mismatch")
    return manifest


def read_safetensors_header(path):
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        require(len(prefix) == 8, f"Truncated safetensors length: {path.name}")
        length = struct.unpack("<Q", prefix)[0]
        require(2 <= length <= 16 * 1024 * 1024 and length <= file_size - 8, f"Invalid safetensors header length: {path.name}")
        raw = handle.read(length)
        require(len(raw) == length, f"Truncated safetensors header: {path.name}")
    header = json.loads(raw)
    require(isinstance(header, dict), f"Safetensors header is not an object: {path.name}")
    offsets, tensors = [], {}
    for name, value in header.items():
        if name == "__metadata__":
            require(isinstance(value, dict), "Invalid safetensors metadata")
            continue
        require(isinstance(value, dict), f"Invalid tensor entry: {name}")
        dtype, shape, position = value.get("dtype"), value.get("shape"), value.get("data_offsets")
        require(dtype in DTYPE_BYTES, f"Unsupported safetensors dtype {dtype}: {name}")
        require(isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape), f"Invalid tensor shape: {name}")
        require(isinstance(position, list) and len(position) == 2 and all(type(n) is int for n in position), f"Invalid tensor offsets: {name}")
        start, end = position
        require(0 <= start <= end <= file_size - 8 - length, f"Tensor outside shard: {name}")
        require(end - start == math.prod(shape) * DTYPE_BYTES[dtype], f"Tensor byte/shape mismatch: {name}")
        offsets.append((start, end, name))
        tensors[name] = value
    require(bool(tensors), f"Empty weight shard: {path.name}")
    cursor = 0
    for start, end, name in sorted(offsets):
        require(start == cursor, f"Tensor data overlaps or contains holes: {name}")
        cursor = end
    require(cursor == file_size - 8 - length, f"Unreferenced shard data: {path.name}")
    return tensors, {"path": path.name, "header_bytes": length, "header_sha256": hashlib.sha256(raw).hexdigest(),
                     "tensors": len(tensors), "data_bytes": cursor}


def validate_model(model_dir, manifest, full_hash=False):
    # Resolve the directory itself, including a link to the existing 475-GiB
    # snapshot. Every source handle is opened read-only; nothing is copied.
    supplied = Path(model_dir).expanduser().absolute()
    resolved = supplied.resolve(strict=True)
    require(resolved.is_dir(), "Model path is not a directory")
    records, shards, all_tensors = [], [], {}
    bytes_hashed = 0
    for entry in manifest["files"]:
        target = resolved / entry["path"]
        require(target.is_file(), f"Missing model file: {entry['path']}")
        require(target.stat().st_size == entry["size"], f"Model file size mismatch: {entry['path']}")
        is_shard = entry["path"].endswith(".safetensors")
        hash_checked = full_hash or not is_shard
        if hash_checked:
            require(hash_file(target) == entry["sha256"], f"Model SHA256 mismatch: {entry['path']}")
            bytes_hashed += entry["size"]
        records.append({"path": entry["path"], "size": entry["size"], "sha256_checked": hash_checked})
        if is_shard:
            tensors, header_record = read_safetensors_header(target)
            header_record["path"] = entry["path"]
            shards.append(header_record)
            for name in tensors:
                require(name not in all_tensors, f"Tensor duplicated across shards: {name}")
                all_tensors[name] = entry["path"]
    config = json.loads((resolved / "config.json").read_text())
    require(config.get("model_type") == "deepseek_v41" and config.get("architectures") == ["DeepseekV41ForCausalLM"], "Wrong model architecture")
    index = json.loads((resolved / "model.safetensors.index.json").read_text())
    mapping = index.get("weight_map")
    require(isinstance(mapping, dict) and mapping == all_tensors, "Shard tensor names/index mapping differ")
    require(set(mapping.values()) == {row["path"] for row in shards}, "Index does not cover all 48 shards")
    metadata = index.get("metadata", {})
    if "total_size" in metadata:
        require(metadata["total_size"] == sum(row["data_bytes"] for row in shards), "Index tensor byte total differs")
    return {"status": "PASS", "supplied_model_dir": str(supplied), "resolved_model_dir": str(resolved),
            "directory_symlink_resolved": str(supplied) != str(resolved), "read_only": True,
            "model_id": MODEL_ID, "revision": MODEL_REVISION, "files_checked": len(records),
            "weight_shards": len(shards), "tensor_count": len(all_tensors), "snapshot_bytes": manifest["snapshot_bytes"],
            "full_hash": full_hash, "bytes_hashed": bytes_hashed,
            "weight_payload_hash_status": "PASS" if full_hash else "NOT_RUN",
            "scope": "All file sizes, all non-shard SHA256, all shard headers and exact index coverage; payload SHA only with full-hash",
            "files": records, "shard_headers": shards}


def available_memory_bytes(meminfo=Path("/proc/meminfo"), cgroup=Path("/sys/fs/cgroup")):
    values = {}
    for line in Path(meminfo).read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key] = int(value.split()[0]) * 1024
    require("MemAvailable" in values, "MemAvailable is missing from /proc/meminfo")
    available = values["MemAvailable"]
    # A container's allocation limit can be below the host's free RAM.
    for limit_name, usage_name in (("memory.max", "memory.current"),
                                   ("memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes")):
        limit_path, usage_path = Path(cgroup) / limit_name, Path(cgroup) / usage_name
        if limit_path.is_file() and usage_path.is_file():
            limit = limit_path.read_text().strip()
            if limit != "max":
                available = min(available, max(0, int(limit) - int(usage_path.read_text().strip())))
    return available


def parse_gpu_csv(text):
    rows = []
    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        require(len(row) == 6, "Unexpected nvidia-smi CSV columns")
        index, uuid, name, capability, total, free = [value.strip() for value in row]
        rows.append({"index": int(index), "uuid": uuid, "name": name, "compute_capability": capability,
                     "memory_total_mib": int(float(total)), "memory_free_mib": int(float(free))})
    return rows


def check_gpu_inventory(rows, expected=8, min_free_vram_mib=76000):
    if not rows:
        raise NoGpu("No visible NVIDIA GPUs; hardware preflight was not run")
    require(len(rows) == expected, f"Expected {expected} GPUs, found {len(rows)}")
    require(len({row["uuid"] for row in rows}) == expected, "Duplicate GPU UUIDs")
    for row in rows:
        require("A100" in row["name"] and row["compute_capability"] == "8.0", f"GPU {row['index']} is not A100 SM80: {row['name']}")
        require(row["memory_total_mib"] >= 80000, f"GPU {row['index']} is not an 80-GB A100")
        require(row["memory_free_mib"] >= min_free_vram_mib,
                f"GPU {row['index']} has {row['memory_free_mib']} MiB free; needs {min_free_vram_mib}. Existing GPU services were not stopped")
    return {"status": "PASS", "devices": rows, "min_free_vram_mib": min_free_vram_mib}


def gpu_preflight(expected=8, min_free_vram_mib=76000):
    executable = shutil.which("nvidia-smi")
    if not executable:
        raise NoGpu("nvidia-smi unavailable; hardware preflight was not run")
    result = subprocess.run([executable, "--query-gpu=index,uuid,name,compute_cap,memory.total,memory.free",
                             "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=30)
    if result.returncode:
        raise NoGpu("nvidia-smi could not query GPUs: " + result.stderr.strip()[:500])
    return check_gpu_inventory(parse_gpu_csv(result.stdout), expected, min_free_vram_mib)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--full-hash", action="store_true", help="Read and SHA256 all 510 GB of weight payloads")
    parser.add_argument("--metadata-only", action="store_true", help="Validate files/headers on CPU; does not accept hardware")
    parser.add_argument("--min-free-vram-mib", type=int, default=76000)
    parser.add_argument("--min-available-ram-gib", type=int, default=512)
    args = parser.parse_args(argv)
    report = {"schema_version": 1, "kind": "deepinfer_project_preflight",
              "mode": "metadata_only" if args.metadata_only else "deployment",
              "created_at": datetime.now(timezone.utc).isoformat(), "status": "NOT_RUN",
              "gpu": {"status": "NOT_RUN"}, "ram": {"status": "NOT_RUN"}}
    exit_code = 1
    try:
        require(args.min_free_vram_mib >= 76000 and args.min_available_ram_gib >= 512, "Resource floors may not be lowered for this package")
        manifest = load_manifest(args.manifest)
        report["manifest_sha256"] = hash_file(args.manifest)
        report["model"] = validate_model(args.model_dir, manifest, args.full_hash)
        if not args.metadata_only:
            # Missing GPUs yield NOT_RUN even on a small build host. Report RAM
            # separately once a GPU inventory exists, without stopping services.
            report["gpu"] = gpu_preflight(8, args.min_free_vram_mib)
            available = available_memory_bytes()
            report["ram"] = {"available_bytes": available, "required_bytes": args.min_available_ram_gib * 1024**3,
                             "status": "PASS" if available >= args.min_available_ram_gib * 1024**3 else "FAIL"}
            require(report["ram"]["status"] == "PASS", f"Available host/cgroup RAM {available / 1024**3:.1f} GiB is below {args.min_available_ram_gib} GiB")
        report["status"] = "PASS"
        exit_code = 0
    except NoGpu as error:
        report.update(status="NOT_RUN", reason=str(error))
        exit_code = 2
    except Exception as error:
        report.update(status="FAIL", reason=f"{type(error).__name__}: {error}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.report)
    print(json.dumps({"status": report["status"], "mode": report["mode"], "reason": report.get("reason"),
                      "gpu": report["gpu"]["status"], "model": report.get("model", {}).get("status", "NOT_RUN")}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
