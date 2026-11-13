"""Launch R8 with original 1017 protocol and PR #66497 compute."""
import json
import os
from pathlib import Path
import sys


def arguments(config, executable=None):
    config=dict(config)
    args=[executable or sys.executable,'-m','adapter.official_server','--model',config.pop('model'),
          '--uds','/state/api.sock','--disable-fastapi-docs','--disable-log-requests']
    for key,value in config.items():
        flag='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:args.append(flag)
            elif key.startswith('enable_'):args.append('--disable-'+key.removeprefix('enable_').replace('_','-'))
            elif key.startswith('disable_'):args.append('--enable-'+key.removeprefix('disable_').replace('_','-'))
            else:args.append('--no-'+key.replace('_','-'))
        else:
            args += [flag,json.dumps(value,separators=(',',':')) if isinstance(value,(dict,list)) else str(value)]
    return args


def main():
    root=Path('/opt/deepinfer')
    config=json.loads((root/'configs/runtime.json').read_text())
    assert config['max_model_len']==524288
    assert config['tensor_parallel_size']==8
    assert config['pipeline_parallel_size']==config['data_parallel_size']==1
    assert config['enable_expert_parallel'] is False
    assert config['speculative_config']['rejection_sample_method']=='standard'
    assert config['speculative_config']['use_local_argmax_reduction'] is False
    assert config['kv_cache_dtype']=='nvfp4_ds_mla'
    assert config['engram_config']['cpu_offload'] is True
    assert config['gpu_memory_utilization']==.94
    assert config['swa_bounded_replay'] is False
    assert config['kv_sharing_fast_prefill'] is False
    assert config['compilation_config']['cudagraph_decoder_replay'] is False
    Path('/state/effective-config.json').write_text(json.dumps(config,indent=2)+'\n')
    os.execv(sys.executable,arguments(config))


if __name__=='__main__':main()
