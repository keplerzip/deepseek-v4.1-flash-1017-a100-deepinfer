"""Start the official-recipe routes using the locked TP8/C32/512K engine."""
import json
import os
from pathlib import Path
import sys

root = Path('/opt/deepinfer')
config = json.loads((root / 'configs/runtime.json').read_text())
if os.environ.get('DEEPINFER_BETA_YOCO_FFN', '0') == '1':
    raise RuntimeError('Approximate YOCO FFN is disabled in the formal CPU KV release')
plan = json.loads(Path('/state/cpu-kv-plan.json').read_text())
if plan['status'] != 'PASS':
    raise RuntimeError('CPU KV memory plan did not pass')
if plan['effective_gib']:
    config['kv_offloading_size'] = plan['effective_gib']
    config['kv_offloading_backend'] = 'native'
    config['disable_hybrid_kv_cache_manager'] = False
    config['kv_transfer_config'] = {
        'kv_connector': 'OffloadingConnector', 'kv_role': 'kv_both',
        'kv_connector_module_path': 'adapter.native_kv_offload',
        'kv_connector_extra_config': {'offload_prompt_only': False,
                                     'blocks_per_chunk': 1, 'eviction_policy': plan['eviction_policy'],
                                     'spec_name': 'RankMajorCPUOffloadingSpec',
                                     'spec_module_path': 'adapter.rank_major_offload'}}
if os.environ.get('DEEPINFER_BETA_EAGER') == '1':
    config['enforce_eager'] = True
    config['compilation_config'] = {'mode': 0, 'cudagraph_mode': 'NONE'}
Path('/state/effective-config.json').write_text(json.dumps(config, indent=2) + '\n')

args = [sys.executable, '-m', 'adapter.official_server', '--model', config.pop('model'), '--uds', '/state/api.sock',
        '--disable-fastapi-docs', '--no-enable-log-requests']
for key, value in config.items():
    flag = '--' + key.replace('_', '-')
    if isinstance(value, bool):
        args.append(flag if value else '--no-' + key.replace('_', '-'))
    else:
        args += [flag, json.dumps(value, separators=(',', ':')) if isinstance(value, (dict, list)) else str(value)]
os.execv(sys.executable, args)
