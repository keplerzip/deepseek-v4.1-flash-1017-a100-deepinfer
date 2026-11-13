"""Build qualification for R8's real recipe/protocol boundary, without any GPU."""
import os
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
env=dict(os.environ,PYTHONPATH=str(root)+':'+str(root/'tests'),DEEPINFER_TOKENIZER=str(root/'vendor/tokenizer/tokenizer.json'))
names=['test_official_server','test_native_token_usage','test_stream_finish_semantics',
       'test_r7_tool_atomic','test_probe_tools','test_port_reuse','test_r8_quality','test_r8_workflow','test_hf2_metadata','test_hf2_update']
subprocess.run([sys.executable,'-m','unittest',*names],cwd=root/'tests',env=env,check=True)
