"""CPU tests for fail-closed evidence binding; no GPU PASS is manufactured."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sm80_gate", ROOT / "scripts/check_sm80_candidate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class CandidateGateTests(unittest.TestCase):
    def test_static_source_integrity_and_safe_defaults(self):
        manifest, checks = gate.static_check()
        self.assertFalse(manifest["default_enabled"])
        self.assertEqual(len(checks), 2)
        self.assertTrue(all(row["syntax"] == "PASS" for row in checks))

    def test_mismatched_vendor_commit_is_rejected(self):
        manifest = gate.read_manifest()
        manifest["source"]["commit"] = "0" * 40
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "commit"):
                gate.read_manifest(path)

    def test_default_opt_in_is_rejected(self):
        manifest = gate.read_manifest()
        manifest["flags"][gate.FLAGS[0]] = "1"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "feature flags"):
                gate.read_manifest(path)

    def test_scalar_oracle_handles_invalid_values_and_request_offset(self):
        result, lengths = gate.reference_combine(
            [[0, -1], [2, 99]], [1, 1], [1, 5], [1, 2], 2, 2, 2, 200, 100)
        self.assertEqual(lengths, [1, 4])
        self.assertEqual(result[0][:5], [100, -1, -1, -1, -1])
        self.assertEqual(result[1][:6], [202, 299, 300, 301, -1, -1])

    def test_scalar_oracle_partial_topk_and_empty(self):
        result, lengths = gate.reference_combine([[5]], [1], [200], [4], 128, 2, 512, 4096, 1024)
        self.assertEqual(lengths, [129])
        self.assertEqual(result[0][0], 5)
        self.assertEqual(result[0][130], -1)
        self.assertEqual(gate.reference_combine([], [], [], [], 128, 2, 512, 1, 1), ([], []))

    def test_full_immutable_image_id_required(self):
        self.assertEqual(gate.validate_image_id("sha256:" + "a" * 64), "sha256:" + "a" * 64)
        for value in (None, "latest", "sha256:abcdef", "image:tag"):
            with self.assertRaises(ValueError):
                gate.validate_image_id(value)

    def test_gpu_evidence_binding_rejects_each_stale_dimension(self):
        manifest = gate.read_manifest()
        image = "sha256:" + "a" * 64
        hashes = {entry["path"]: entry["patched_sha256"] for entry in manifest["files"]}
        devices = [{"uuid": "GPU-fixture-only", "capability": [8, 0]}]
        report = {"schema_version": 1, "kind": "sm80_candidate_gate", "mode": "gpu", "status": "PASS",
                  "manifest_sha256": gate.sha256(gate.MANIFEST), "base_image_id": manifest["base_image_id"],
                  "image_id": image, "installed_files": hashes, "gpu_devices": devices,
                  "validated_candidates": sorted(gate.REQUIRED_CANDIDATES), "checks": {"gpu": [{"status": "PASS"}]}}
        gate.evidence_binding(report, manifest, hashes, image, devices)
        # These dictionaries exist only in a temporary Python test object.
        # No fixture may be written to validation/ or accepted by startup.
        for key, value in (("mode", "static"), ("status", "NOT_RUN"), ("manifest_sha256", "0" * 64),
                           ("base_image_id", "wrong"), ("image_id", "wrong"), ("installed_files", {}),
                           ("gpu_devices", []), ("validated_candidates", ["topk_swa"]), ("checks", {"gpu": []})):
            stale = copy.deepcopy(report)
            stale[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                gate.evidence_binding(stale, manifest, hashes, image, devices)

    def test_installed_module_hash_mismatch_rejected(self):
        manifest = gate.read_manifest()
        with tempfile.TemporaryDirectory() as directory:
            for entry in manifest["files"]:
                target = Path(directory) / entry["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("# incompatible source\n")
            with self.assertRaisesRegex(ValueError, "Installed hash mismatch"):
                gate.installed_hashes(manifest, directory)

    def test_no_torch_is_not_run(self):
        class Cuda:
            @staticmethod
            def is_available():
                return False
        class Torch:
            cuda = Cuda()
        with self.assertRaises(gate.NotRun):
            gate.gpu_inventory(Torch, 8)


if __name__ == "__main__":
    unittest.main()
