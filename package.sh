#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
[[ -f "$ROOT/images/runtime-image.tar" ]] || fail 'Run export-runtime.sh first'
mkdir -p "$ROOT/dist"
python3 "$ROOT/scripts/package_manifest.py" "$ROOT"
parent=$(dirname -- "$ROOT")
name=$(basename -- "$ROOT")
archive="$ROOT/dist/DeepSeek-Flash-1017-A100-DeepInfer-20261022-R3-CPUKV-runtime.tar.gz"
if command -v pigz >/dev/null; then COMPRESS=(pigz -p 8 -6); else COMPRESS=(gzip -6); fi
tar --sort=name --mtime='UTC 2026-10-21 16:00:00' --owner=0 --group=0 --numeric-owner \
  --exclude="$name/dist" --exclude="$name/runtime" --exclude="$name/.git" \
  --exclude='__pycache__' --exclude='*.pyc' -C "$parent" -cf - "$name" | "${COMPRESS[@]}" > "$archive.partial"
mv -- "$archive.partial" "$archive"
cd "$ROOT/dist"
sha256sum -- "$(basename -- "$archive")" > "$(basename -- "$archive").sha256"
printf 'Offline runtime package: %s\n' "$archive"
