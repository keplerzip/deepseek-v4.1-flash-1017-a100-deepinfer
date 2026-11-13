"""Verify every saved image blob and bind portable index/manifest/config IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile


def inspect_archive(path, tag, expected_id):
    identities, checked, total = set(), 0, 0
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        for member in members:
            if member.issym() or member.islnk() or member.name.startswith('/') or '..' in Path(member.name).parts:
                raise ValueError('Unsafe image archive member')
            if member.isfile() and member.name.startswith('blobs/sha256/'):
                digest = hashlib.sha256()
                with archive.extractfile(member) as stream:
                    for data in iter(lambda: stream.read(16 * 1024**2), b''):
                        digest.update(data)
                if digest.hexdigest() != Path(member.name).name:
                    raise ValueError('Image blob checksum mismatch: ' + member.name)
                checked += 1
                total += member.size
        manifests = json.load(archive.extractfile('manifest.json'))
        selected = [item for item in manifests if tag in item['RepoTags']]
        if len(selected) != 1:
            raise ValueError('Saved image tag is missing or ambiguous')
        manifest = selected[0]
        config_name = manifest['Config']
        identities.add('sha256:' + Path(config_name).stem)
        config = json.load(archive.extractfile(config_name))
        if config.get('architecture') != 'amd64' or config.get('os') != 'linux':
            raise ValueError('Expected linux/amd64 runtime')
        for name in manifest['Layers']:
            if not archive.getmember(name).isfile():
                raise ValueError('Missing saved layer')
        if 'index.json' in archive.getnames():
            def visit(descriptor):
                digest = descriptor['digest']
                media = descriptor.get('mediaType', '')
                if 'index' in media or 'manifest' in media:
                    data = json.load(archive.extractfile('blobs/sha256/' + digest.split(':')[1]))
                    if 'manifests' in data:
                        matched = False
                        for child in data['manifests']:
                            if child.get('platform', {}).get('architecture') in (None, 'amd64'):
                                matched = visit(child) or matched
                        if matched:
                            identities.add(digest)
                        return matched
                    elif data.get('config', {}).get('digest') == 'sha256:' + Path(config_name).stem:
                        identities.add(digest)
                        return True
                return False
            for descriptor in json.load(archive.extractfile('index.json'))['manifests']:
                visit(descriptor)
        if expected_id not in identities:
            raise ValueError('Built image identity is absent from saved archive')
    return {'status': 'PASS', 'image_tag': tag, 'build_image_id': expected_id,
            'accepted_image_ids': sorted(identities), 'verified_blobs': checked,
            'verified_blob_bytes': total, 'layers': len(manifest['Layers']),
            'archive_bytes': Path(path).stat().st_size, 'weights_included': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--env', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', args.image_id):
        raise ValueError('Invalid immutable image ID')
    report = inspect_archive(args.archive, args.tag, args.image_id)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    args.env.write_text(f'IMAGE_TAG={args.tag}\nIMAGE_ID={args.image_id}\nIMAGE_ALLOWED_IDS="' + ' '.join(report['accepted_image_ids']) + '"\n')
    print(json.dumps(report, indent=2))
