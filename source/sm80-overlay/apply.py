#!/usr/bin/env python3
"""Reconstruct selected functions into a separate, SHA-bound overlay directory."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def function_block(source: str, name: str) -> str:
    matches = [node for node in ast.parse(source).body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one top-level function: {name}")
    node = matches[0]
    start = min([node.lineno] + [item.lineno for item in node.decorator_list])
    return "\n".join(source.splitlines()[start - 1:node.end_lineno]) + "\n"


def prepare(source_root: Path, output_root: Path | None = None) -> dict:
    source_root = source_root.resolve(strict=True)
    if output_root is not None:
        output_root = output_root.resolve()
        if (output_root == source_root or output_root.is_relative_to(source_root)
                or source_root.is_relative_to(output_root)):
            raise ValueError("The output must be separate from the source tree")
    manifest = json.loads((ROOT / "manifest.json").read_text())
    pending = []
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe manifest path")
        source_path = source_root / relative
        if source_path.is_symlink():
            raise ValueError(f"Refuse symlink source: {source_path}")
        raw = source_path.read_bytes()
        if digest(raw) != entry["base_sha256"]:
            raise ValueError(f"Source differs from the pinned R2 image: {relative}")
        fragment_path = ROOT / entry["fragment"]
        fragment = fragment_path.read_bytes()
        if digest(fragment) != entry["fragment_sha256"]:
            raise ValueError(f"Fragment checksum mismatch: {entry['fragment']}")
        if digest((ROOT / entry["review_patch"]).read_bytes()) != entry["patch_sha256"]:
            raise ValueError(f"Review patch checksum mismatch: {entry['review_patch']}")
        patched = raw.decode()
        for replacement in entry["replacements"]:
            original = function_block(patched, replacement["original"])
            blocks = []
            if replacement.get("preserve_as"):
                blocks.append(original.replace(
                    "def " + replacement["original"] + "(",
                    "def " + replacement["preserve_as"] + "(", 1))
            blocks.extend(function_block(fragment.decode(), name)
                          for name in replacement["insert"])
            patched = patched.replace(original, "\n\n".join(blocks), 1)
        compile(patched, str(relative), "exec")
        result = patched.encode()
        if digest(result) != entry["patched_sha256"]:
            raise ValueError(f"Reconstructed output checksum mismatch: {relative}")
        pending.append((relative, result, entry))
    # Validate every source and output before writing either generated module.
    for relative, raw, _ in pending:
        if output_root is not None:
            target = output_root / relative
            if target.exists() and (target.is_symlink() or target.read_bytes() != raw):
                raise ValueError(f"Refuse to replace a different output: {target}")
            ancestor = target.parent
            while not ancestor.exists():
                ancestor = ancestor.parent
            if not ancestor.resolve().is_relative_to(output_root) and ancestor != output_root.parent:
                # An existing symlink may not redirect an output parent.
                if any(parent.is_symlink() for parent in target.parents):
                    raise ValueError("Output parent escapes through a symlink")
    for relative, raw, _ in pending:
        if output_root is not None:
            target = output_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.parent.resolve().is_relative_to(output_root):
                raise ValueError("Output parent escapes through a symlink")
            target.write_bytes(raw)
    return {
        "status": "PASS",
        "scope": "Exact R2 source hashes, selected AST edits, Python syntax and output hashes only",
        "base_image_id": manifest["base_image_id"],
        "source_modified": False,
        "gpu_executed": False,
        "gpu_validated": False,
        "flags_default": manifest["flags"],
        "output_root": str(output_root) if output_root is not None else None,
        "files": [{"path": str(relative), "base_sha256": entry["base_sha256"],
                   "patched_sha256": digest(raw)} for relative, raw, entry in pending],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Directory containing the exact R2 vllm package")
    parser.add_argument("--output-root", type=Path,
                        help="Separate directory to receive two generated modules")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.output_root is None and not args.check_only:
        parser.error("Choose --output-root or --check-only")
    try:
        report = prepare(args.source_root, None if args.check_only else args.output_root)
        code = 0
    except (OSError, ValueError, SyntaxError, KeyError) as error:
        report = {"status": "FAIL", "gpu_validated": False,
                  "error": f"{type(error).__name__}: {error}"}
        code = 1
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
