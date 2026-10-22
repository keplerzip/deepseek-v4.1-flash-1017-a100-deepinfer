import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("preflight", ROOT / "scripts/preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def write_shard(path, name="weight"):
    header = json.dumps({name: {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00\x00\x01\x00")


class PreflightTests(unittest.TestCase):
    def model_fixture(self, root):
        write_shard(root / "model.safetensors")
        (root / "config.json").write_text(json.dumps({"model_type": "deepseek_v41", "architectures": ["DeepseekV41ForCausalLM"]}))
        (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 4}, "weight_map": {"weight": "model.safetensors"}}))
        files = [{"path": path.name, "size": path.stat().st_size, "sha256": preflight.hash_file(path)} for path in sorted(root.iterdir())]
        return {"files": files, "snapshot_bytes": sum(entry["size"] for entry in files)}

    def test_real_manifest_identity_and_complete_snapshot(self):
        manifest = preflight.load_manifest()
        self.assertEqual(len(manifest["files"]), 84)
        self.assertEqual(sum(row["path"].endswith(".safetensors") for row in manifest["files"]), 48)
        self.assertEqual(manifest["snapshot_bytes"], 510311045870)

    def test_directory_symlink_reuses_model_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model"
            root.mkdir()
            manifest = self.model_fixture(root)
            link = Path(directory) / "reused-model"
            link.symlink_to(root, target_is_directory=True)
            before = {path.name: (path.stat().st_mtime_ns, path.stat().st_ino) for path in root.iterdir()}
            result = preflight.validate_model(link, manifest)
            self.assertTrue(result["directory_symlink_resolved"])
            self.assertEqual(result["resolved_model_dir"], str(root.resolve()))
            self.assertEqual(result["weight_payload_hash_status"], "NOT_RUN")
            self.assertEqual(before, {path.name: (path.stat().st_mtime_ns, path.stat().st_ino) for path in root.iterdir()})

    def test_default_does_not_claim_payload_sha_full_hash_detects_damage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.model_fixture(root)
            shard = root / "model.safetensors"
            data = bytearray(shard.read_bytes())
            data[-1] ^= 1
            shard.write_bytes(data)
            self.assertEqual(preflight.validate_model(root, manifest)["status"], "PASS")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                preflight.validate_model(root, manifest, full_hash=True)

    def test_small_file_tamper_is_rejected_without_full_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.model_fixture(root)
            path = root / "config.json"
            path.write_text(path.read_text().replace("deepseek_v41", "deepseek_v42"))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                preflight.validate_model(root, manifest)

    def test_header_shape_byte_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.safetensors"
            header = json.dumps({"x": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 4]}}).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"0000")
            with self.assertRaisesRegex(ValueError, "byte/shape mismatch"):
                preflight.read_safetensors_header(path)

    def test_index_to_shard_reference_must_match_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.model_fixture(root)
            path = root / "model.safetensors.index.json"
            obj = json.loads(path.read_text())
            obj["weight_map"] = {"other": "model.safetensors"}
            path.write_text(json.dumps(obj))
            for entry in manifest["files"]:
                if entry["path"] == path.name:
                    entry.update(size=path.stat().st_size, sha256=preflight.hash_file(path))
            with self.assertRaisesRegex(ValueError, "index mapping"):
                preflight.validate_model(root, manifest)

    def test_gpu_inventory_fails_on_occupied_card_without_mutation(self):
        line = "0, GPU-0, NVIDIA A100-SXM4-80GB, 8.0, 81920, 81000"
        rows = preflight.parse_gpu_csv(line)
        self.assertEqual(preflight.check_gpu_inventory(rows, expected=1)["status"], "PASS")
        rows[0]["memory_free_mib"] = 5000
        with self.assertRaisesRegex(ValueError, "Existing GPU services were not stopped"):
            preflight.check_gpu_inventory(rows, expected=1)
        self.assertEqual(rows[0]["memory_free_mib"], 5000)

    def test_wrong_card_or_memory_is_rejected(self):
        for line in ("0, GPU-0, NVIDIA H100, 9.0, 81920, 81000", "0, GPU-0, NVIDIA A100, 8.0, 40960, 40000"):
            with self.assertRaises(ValueError):
                preflight.check_gpu_inventory(preflight.parse_gpu_csv(line), expected=1)
        with self.assertRaises(preflight.NoGpu):
            preflight.check_gpu_inventory([])

    def test_available_ram_respects_cgroup_headroom(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = root / "meminfo"
            info.write_text("MemAvailable: 1099511627 kB\n")
            (root / "memory.max").write_text(str(700 * 1024**3))
            (root / "memory.current").write_text(str(300 * 1024**3))
            self.assertEqual(preflight.available_memory_bytes(info, root), 400 * 1024**3)

    def test_deployment_missing_gpu_is_not_run_exit_two(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            with mock.patch.object(preflight, "load_manifest", return_value={}), \
                 mock.patch.object(preflight, "hash_file", return_value="a" * 64), \
                 mock.patch.object(preflight, "validate_model", return_value={"status": "PASS"}), \
                 mock.patch.object(preflight, "gpu_preflight", side_effect=preflight.NoGpu("No actual GPU")):
                code = preflight.main(["--model-dir", str(root), "--report", str(report)])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(report.read_text())["status"], "NOT_RUN")


if __name__ == "__main__":
    unittest.main()
