"""Offline SM80 compilation or optional real GPU numeric/graph A/B checks."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def standalone_module():
    path = ROOT / 'source/beta-overlay/files/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py'
    source = path.read_text()
    names = {'_decode_e8m0_scales_triton', '_decode_e8m0_scales_bf16_triton',
             '_sparse_attn_decode_partial_kernel', '_beta_reference_decode_partial_kernel'}
    chunks = []
    # vLLM intentionally disables its Triton wrapper on a host without a
    # CUDA driver. Extract the identical FP8 LUT helper and compile with real
    # Triton directly, as in the existing offline combine compiler.
    import importlib.util
    package = importlib.util.find_spec('vllm')
    fp8_source = (Path(package.origin).parent / 'v1/attention/ops/fp8_sm80.py').read_text()
    for n in ast.parse(fp8_source).body:
        if isinstance(n, ast.FunctionDef) and n.name == '_decode_fp8_lut':
            start = min([n.lineno] + [d.lineno for d in n.decorator_list]) - 1
            chunks.append('\n'.join(fp8_source.splitlines()[start:n.end_lineno]))
    for n in ast.parse(source).body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            start = min([n.lineno] + [d.lineno for d in n.decorator_list]) - 1
            chunks.append('\n'.join(source.splitlines()[start:n.end_lineno]))
    temp = tempfile.TemporaryDirectory()
    file = Path(temp.name) / 'beta_kernels.py'
    file.write_text('import triton\nimport triton.language as tl\n'
                    '_NATIVE_FP8_CAST = tl.constexpr(False)\n\n' + '\n\n'.join(chunks))
    spec = importlib.util.spec_from_file_location('deepinfer_beta_kernels', file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, temp


def compile_checks():
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    module, temp = standalone_module()
    records = []
    try:
        kernel = module._sparse_attn_decode_partial_kernel
        for splits, extra, warps in [(1,False,8), (4,True,4), (8,True,4), (16,True,8)]:
            constants = dict(HAS_EXTRA=extra, NOPE_DIM=448, NOPE_BLOCK=512, ROPE_DIM=64,
                             IS_FNUZ_MAIN=False, IS_FNUZ_EXTRA=False, BLOCK_H=16,
                             BLOCK_K=32, NUM_SPLITS=splits, NUM_STAGES=1)
            signature = {}
            for name in kernel.arg_names:
                if name in constants:
                    signature[name] = 'constexpr'
                elif name.endswith('_ptr'):
                    signature[name] = ('*bf16' if name in ('q_ptr','fp8_lut_ptr') else
                                       '*u8' if 'cache' in name else
                                       '*fp32' if name.startswith('part_') else '*i32')
                else:
                    signature[name] = 'fp32' if name == 'scale' else 'i32'
            result = triton.compile(ASTSource(kernel, signature, constants), target=GPUTarget('cuda',80,32),
                                    options={'num_warps':warps})
            records.append({'splits':splits,'extra':extra,'warps':warps,
                            'shared_memory':result.metadata.shared, 'formats':list(result.asm),'status':'PASS'})
    finally:
        temp.cleanup()
    return records


def gpu_checks():
    import os
    import torch
    if not torch.cuda.is_available():
        return None
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops
    records = []
    for gpu in range(torch.cuda.device_count()):
        torch.cuda.set_device(gpu)
        device = torch.device('cuda',gpu)
        if torch.cuda.get_device_capability(device) != (8,0):
            raise RuntimeError('GPU check is scoped to SM80')
        generator = torch.Generator().manual_seed(20261021)
        blocks, block = 8,64
        raw = torch.zeros((blocks,block*584),dtype=torch.uint8)
        for i in range(blocks):
            data = raw[i,:block*576].view(block,576)
            data[:,:448] = torch.randn((block,448),generator=generator).to(torch.float8_e4m3fn).view(torch.uint8)
            data[:,448:] = torch.randn((block,64),generator=generator).bfloat16().view(torch.uint8)
            raw[i,block*576:] = torch.randint(124,130,(block*8,),generator=generator,dtype=torch.uint8)
        cache = raw.view(blocks,block,584).to(device)
        for queries in (1,6,12,192):
            print(json.dumps({"stage":"mla_gpu_case","gpu":gpu,"queries":queries}),flush=True)
            q = torch.randn((queries,8,512),generator=generator).bfloat16().to(device)*.05
            main = torch.randint(0,blocks*block,(queries*128,),generator=generator,dtype=torch.int32).to(device)
            extra = torch.randint(0,blocks*block,(queries*512,),generator=generator,dtype=torch.int32).to(device)
            mainptr = torch.arange(queries+1,device=device,dtype=torch.int32)*128
            extraptr = torch.arange(queries+1,device=device,dtype=torch.int32)*512
            args = (q,cache,main,mainptr,512**-.5,None,448,64,cache,extra,extraptr)
            os.environ['DEEPINFER_BETA_UNIFIED_MLA']='0'
            reference = ops._rocm_sparse_attn_decode_ragged_triton(*args)
            os.environ['DEEPINFER_BETA_UNIFIED_MLA']='1'
            actual = ops._rocm_sparse_attn_decode_ragged_triton(*args)
            torch.cuda.synchronize(device)
            torch.testing.assert_close(actual,reference,rtol=.03,atol=.02)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=torch.cuda.Stream(device=device)):
                captured = ops._rocm_sparse_attn_decode_ragged_triton(*args)
            graph.replay()
            graph.replay()
            torch.cuda.synchronize(device)
            torch.testing.assert_close(captured,actual,rtol=0,atol=0)
            records.append({'gpu':gpu,'queries':queries,'heads':8,'max_abs_error':float((actual-reference).abs().max()),
                            'cuda_graph_replayed':True,'status':'PASS'})
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode',choices=('compile','gpu'),default='compile')
    p.add_argument('--report',type=Path,required=True)
    args = p.parse_args()
    report = {'mode':args.mode,'status':'NOT_RUN','gpu_executed':False,
              'scope':'MLA kernel only; excludes Engram graph overlap, model quality and serving capacity'}
    code = 0
    try:
        checks = compile_checks() if args.mode == 'compile' else gpu_checks()
        if checks is None:
            report['reason'] = 'No CUDA GPU is available'
            code = 2
        else:
            report.update(status='PASS',checks=checks,gpu_executed=args.mode=='gpu')
    except Exception as e:
        import traceback
        traceback.print_exc()
        report.update(status='FAIL',error=f'{type(e).__name__}: {e}')
        code = 1
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
