"""Hash source/runtime package files, excluding only transient and output state."""
import argparse
import hashlib
from pathlib import Path


EXCLUDED = {'dist', 'runtime', '.git', '__pycache__'}


def included(path, root):
    relative = path.relative_to(root)
    return not any(part in EXCLUDED for part in relative.parts) and path.name != 'SHA256SUMS' and path.suffix != '.pyc'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = []
    for path in sorted(root.rglob('*')):
        if not included(path, root):
            continue
        if path.is_symlink():
            raise ValueError('Package must not depend on external symlinks: ' + str(path))
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for data in iter(lambda: stream.read(16 * 1024**2), b''):
                digest.update(data)
        rows.append(digest.hexdigest() + '  ' + path.relative_to(root).as_posix())
    (root / 'SHA256SUMS').write_text('\n'.join(rows) + '\n')
    print(f'Hashed {len(rows)} package files')
