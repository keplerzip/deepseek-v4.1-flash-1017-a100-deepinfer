#!/usr/bin/env python3
"""Validate the complete offline tar, exact membership, hashes and gzip trailer."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

ROOT = 'deepseek-v4.1-flash-1017-a100-deepinfer'
EXCLUDED = {'.git', '.build', 'runtime', 'dist', '__pycache__'}
WEIGHTS = {'.safetensors', '.gguf', '.pt', '.pth', '.bin'}
BLOCK = 16 * 1024 * 1024


def verify(path, expected_root=ROOT):
    actual, expected, seen = {}, None, set()
    total = 0
    with open(path, 'rb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='rb') as gz:
            with tarfile.open(fileobj=gz, mode='r|') as archive:
                for member in archive:
                    name = PurePosixPath(member.name)
                    if not name.parts or name.is_absolute() or '..' in name.parts or name.parts[0] != expected_root:
                        raise ValueError('Unsafe or unexpected root: ' + member.name)
                    canonical = name.as_posix()
                    if canonical in seen:
                        raise ValueError('Duplicate member: ' + member.name)
                    seen.add(canonical)
                    if member.uid != 0 or member.gid != 0:
                        raise ValueError('Nonzero tar owner: ' + member.name)
                    relparts = name.parts[1:]
                    if any(p in EXCLUDED for p in relparts):
                        raise ValueError('Excluded path: ' + member.name)
                    if member.isdir():
                        continue
                    if not member.isfile() or not relparts:
                        raise ValueError('Unsupported archive member: ' + member.name)
                    relative = PurePosixPath(*relparts).as_posix()
                    if name.suffix.lower() in WEIGHTS or name.suffix == '.pyc':
                        raise ValueError('Forbidden model or cache file: ' + member.name)
                    if relative == 'SHA256SUMS' and member.size > 5 * 1024 * 1024:
                        raise ValueError('Unreasonably large SHA256SUMS')
                    hasher = hashlib.sha256()
                    chunks = [] if relative == 'SHA256SUMS' else None
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError('Missing file stream')
                    while data := stream.read(BLOCK):
                        hasher.update(data)
                        if chunks is not None:
                            chunks.append(data)
                    total += member.size
                    if chunks is not None:
                        expected = {}
                        for line in b''.join(chunks).decode().splitlines():
                            digest, filename = line.split('  ', 1)
                            filename_path = PurePosixPath(filename)
                            if not re.fullmatch('[0-9a-f]{64}', digest) or filename in expected or filename_path.is_absolute() or '..' in filename_path.parts:
                                raise ValueError('Invalid manifest entry: ' + line)
                            expected[filename] = digest
                    else:
                        actual[relative] = hasher.hexdigest()
                # Drain through tarfile's stream adapter, including its read-
                # ahead buffer. Only zero tar padding may follow end markers;
                # reject appended tar/gzip payloads hidden after the first EOA.
                while tail := archive.fileobj.read(BLOCK):
                    if any(tail):
                        raise ValueError('Nonzero data after tar end markers')
            # tarfile stops at end-of-archive blocks; drain gzip to validate CRC and length.
            while gz.read(BLOCK):
                pass
        if raw.read(1):
            raise ValueError('Unexpected bytes after gzip stream')
    if expected is None or actual != expected:
        missing = sorted(set(expected or {}) - set(actual))[:10]
        extra = sorted(set(actual) - set(expected or {}))[:10]
        mismatched = sorted(k for k in actual.keys() & (expected or {}).keys() if actual[k] != expected[k])[:10]
        raise ValueError(f'Manifest mismatch missing={missing} extra={extra} changed={mismatched}')
    if 'images/runtime-image.tar' not in actual:
        raise ValueError('Offline image is missing')
    with open(path, 'rb') as f:
        outer = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'status':'PASS','root':expected_root,'verified_files':len(actual),
            'uncompressed_file_bytes':total,'compressed_bytes':Path(path).stat().st_size,
            'sha256':outer,'gzip_crc':'PASS','membership_and_checksums':'PASS',
            'weights_included':False,'target_gpu_tested':False}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('archive', type=Path)
    p.add_argument('--report', type=Path)
    p.add_argument('--root', default=ROOT)
    a = p.parse_args()
    report = verify(a.archive, a.root)
    if a.report:
        a.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
