"""Read the complete gzip tar and verify every packaged file against SHA256SUMS."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def verify(path):
    actual, expected, top, size = {}, None, None, 0
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or member.issym() or member.islnk():
                raise ValueError('Unsafe archive member: ' + member.name)
            if top is None:
                top = name.parts[0]
            if name.parts[0] != top:
                raise ValueError('Multiple archive roots')
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError('Unsupported member type')
            relative = str(PurePosixPath(*name.parts[1:]))
            if relative in actual:
                raise ValueError('Duplicate archive file')
            digest = hashlib.sha256()
            manifest_data = bytearray() if relative == 'SHA256SUMS' else None
            with archive.extractfile(member) as stream:
                for data in iter(lambda: stream.read(16 * 1024**2), b''):
                    digest.update(data)
                    if manifest_data is not None:
                        manifest_data.extend(data)
            size += member.size
            if manifest_data is not None:
                expected = {}
                for line in manifest_data.decode().splitlines():
                    checksum, filename = line.split('  ', 1)
                    if filename in expected:
                        raise ValueError('Duplicate checksum manifest entry')
                    expected[filename] = checksum
            else:
                actual[relative] = digest.hexdigest()
    if expected is None or actual != expected:
        raise ValueError('Archive file checksums or membership differ from manifest')
    if 'images/runtime-image.tar' not in actual:
        raise ValueError('Complete offline image is absent')
    if any(filename.endswith('.safetensors') for filename in actual):
        raise ValueError('Model weights must be external to this runtime package')
    return {'status': 'PASS', 'root': top, 'verified_files': len(actual),
            'uncompressed_file_bytes': size, 'compressed_bytes': Path(path).stat().st_size,
            'complete_gzip_stream_read': True, 'file_checksums': 'PASS', 'weights_included': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.archive)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
