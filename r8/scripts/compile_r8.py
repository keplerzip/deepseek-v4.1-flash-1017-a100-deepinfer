"""Cross-compile actual PR kernels to SM80 without allocating or executing CUDA.

Mock only the device-context selector/properties needed by host compilation;
CuTe compiler, PTX assembler and installed upstream kernel bodies remain real.
A compile PASS is not numerical or serving validation.
"""
import argparse
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
from unittest.mock import patch
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);args=p.parse_args()
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import (
        quantize, sparse_mla, topk, mqa_logits, mhc, mhc_projection, gemv, all_reduce, all_gather)
    work=[]
    def add(name,fn,*a):work.append((name,fn,a))
    for dtype in (torch.float32,torch.bfloat16):add('indexer-query/'+str(dtype),quantize._query_kernel,0,dtype)
    for dtype in (torch.int32,torch.int64):
        add('indexer-key/'+str(dtype),quantize._key_kernel,0,dtype)
        add('nvfp4-compressed-kv/'+str(dtype),quantize._record_kernel,0,dtype)
    for compressed in (False,True):add('sparse-mla/8-heads/'+str(compressed),sparse_mla._partial_kernel,0,compressed,8,False,True)
    add('sparse-mla/merge',sparse_mla._merge_kernel,0,True)
    for k,group,remap,stage in [(512,1,0,0),(512,1,0,1),(512,1,0,2),(512,1,8,0),
                               (2048,8,0,0),(2048,8,0,1),(2048,8,0,2)]:
        add(f'topk/{k}/{group}/{remap}/{stage}',topk._kernel,0,k,group,remap,stage)
    for paged in (False,True):
        for candidate in (0,8):
            for group in ((1,6) if paged else (1,)):
                if candidate and (not paged or group != 1):continue
                add(f'indexer-logits/{paged}/{candidate}/{group}',mqa_logits._kernel,0,paged,32,candidate,group,False)
    for has_post in (False,True):
        for aux in (False,True):add(f'mhc/post/{has_post}/{aux}',mhc._post_collapse_kernel,0,5120,has_post,aux)
        add(f'mhc/coeff/{has_post}',mhc._coefficients_mma_kernel,0,5120,has_post)
    add('mhc/prefill-projection',mhc_projection._kernel,0,20480)
    add('all-reduce/TP8',all_reduce._kernel,0,5120,8,128)
    for aux in (False,True):
        add(f'all-reduce/post/{aux}',all_reduce._collapse_kernel,0,5120,8,128,aux)
        add(f'all-reduce/finalize/{aux}',all_reduce._finalize_kernel,0,5120,8,128,aux,6)
    for fragments,capacity,side in [(128,16,False),(2020,128,True),(8,128,True),(32,128,True),(40,128,True)]:
        add(f'all-gather/{fragments}/{side}',all_gather._kernel,0,fragments,8,capacity,side)
    for n,k in [(1280,5120),(512,5120),(1280,1280),(1024,5120),(5120,1024),(1536,5120)]:
        for rows in (6,12,24):
            cfg=gemv.default_config(n,k,rows,torch.device('cuda:0'))
            add(f'gemv/{n}/{k}/{rows}',gemv._kernel,0,n,k,(rows+7)//8,cfg)
    result=dict(status='RUNNING',scope='Actual SM80 cross compilation only',gpu_executed=False,
        numerical_validation='NOT_RUN',full_model_quality='NOT_RUN',cases=[])
    args.report.parent.mkdir(parents=True,exist_ok=True)
    for name,fn,a in work:
        start=time.monotonic()
        try:
            compiled=fn(*a)
            artifacts={}
            for suffix,attribute in [('ptx','__ptx__'),('cubin','__cubin__')]:
                value=getattr(compiled,attribute,None)
                data=None
                if isinstance(value,bytes):data=value
                elif isinstance(value,str):
                    if len(value)<4096 and '\n' not in value and Path(value).is_file():
                        data=Path(value).read_bytes()
                    elif suffix=='ptx' and '.target sm_80' in value:
                        data=value.encode()
                if data:
                    if suffix=='cubin' and not data.startswith(b'\x7fELF'):
                        raise RuntimeError('Assembler output is not an ELF CUBIN')
                    destination=args.report.parent/'sm80-artifacts'/(name.replace('/','_')+'.'+suffix)
                    destination.parent.mkdir(exist_ok=True);destination.write_bytes(data)
                    import hashlib
                    artifacts[suffix]={'file':destination.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            if not artifacts:raise RuntimeError('Compiler emitted no retained PTX/CUBIN artifact')
            row=dict(name=name,status='PASS',seconds=time.monotonic()-start,artifacts=artifacts)
        except Exception as exc:row=dict(name=name,status='FAIL',error=type(exc).__name__+': '+str(exc)[:400])
        result['cases'].append(row);args.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(row),flush=True)
    result['status']='PASS' if all(r['status']=='PASS' for r in result['cases']) else 'FAIL'
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['status']=='PASS' else 1

if __name__=='__main__':
    from cuda.bindings import driver
    def attribute(attr,device):
        if attr != driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR:
            raise RuntimeError('Unexpected hardware-dependent compile query: '+str(attr))
        return driver.CUresult.CUDA_SUCCESS,163840
    props=SimpleNamespace(multi_processor_count=108,shared_memory_per_block_optin=163840)
    with patch.object(torch.accelerator,'device_index',lambda _:contextlib.nullcontext()), \
         patch.object(torch.cuda,'get_device_properties',lambda _:props), \
         patch.object(driver,'cuDeviceGet',lambda *a:(driver.CUresult.CUDA_SUCCESS,0)), \
         patch.object(driver,'cuDeviceGetAttribute',attribute):
        raise SystemExit(main())
