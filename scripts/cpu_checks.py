"""Run bundled checks without network/GPU, including real Triton interpretation."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
env = dict(os.environ, PYTHONPATH=str(root), DEEPINFER_TOKENIZER=str(root / 'vendor/tokenizer/tokenizer.json'),
           DEEPINFER_R2_SOURCE=str(root / 'vendor/r2-source'))
for directory, additions in [(root / 'tests', {}),
                             (root / 'source/sm80-overlay/tests', {'TRITON_INTERPRET': '1'})]:
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(directory), '-p', 'test_*.py'],
                   env=env | additions, check=True)
