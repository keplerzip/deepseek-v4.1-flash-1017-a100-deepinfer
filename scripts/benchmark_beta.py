"""Opt-in local HTTP load/long-context measurement; writes actual usage and TTFT."""
import argparse
import asyncio
import json
from pathlib import Path
import time

import httpx
import deepseek_recipe as recipe
from tokenizers import Tokenizer as TextTokenizer


def build_filler(path, prompt_tokens):
    if prompt_tokens < 1:
        raise ValueError('Prompt token budget must be positive')
    tokenizer = recipe.Tokenizer.from_file(str(path))
    unit = 'This is reference material for a long-context inference test.\n'
    encoded = tokenizer.encode(unit * (prompt_tokens // 8 + 1))[:prompt_tokens]
    # recipe.Tokenizer exposes encode, not decode. Both readers use the same
    # checkpoint tokenizer.json; decode only the benchmark's plain ASCII filler.
    filler = TextTokenizer.from_file(str(path)).decode(encoded, skip_special_tokens=False)
    actual = len(tokenizer.encode(filler))
    if actual != prompt_tokens:
        raise ValueError(f'Filler token count {actual} does not match requested {prompt_tokens}')
    return filler


async def run(args):
    filler = build_filler(args.tokenizer, args.prompt_tokens)
    semaphore = asyncio.Semaphore(args.concurrency)
    transport = httpx.AsyncHTTPTransport(uds=args.uds) if args.uds else None
    async with httpx.AsyncClient(timeout=args.timeout, transport=transport) as client:
        info = (await client.get(args.url.rstrip('/') + '/deepinfer/compatibility')).json()
        async def one(index):
            async with semaphore:
                marker = f'BETA_NEEDLE_{index:04d}_9f3a'
                start = time.perf_counter()
                ttft, final, text = None, None, ''
                try:
                    body = {'model':'DeepSeek-Flash', 'reasoning':{'effort':'none'}, 'stream':True,
                            'max_output_tokens':args.output_tokens,
                            'input':f'The secret marker is {marker}.\n{filler}\nRepeat the secret marker from the beginning, then briefly explain it.'}
                    async with client.stream('POST',args.url.rstrip('/')+'/v1/responses',json=body) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line.startswith('data: '):
                                continue
                            event = json.loads(line[6:])
                            if event['type'] == 'response.output_text.delta':
                                if ttft is None:
                                    ttft = time.perf_counter()-start
                                text += event['delta']
                            if event['type'] in ('response.completed','response.incomplete','response.failed'):
                                final = event['response']
                                break
                    elapsed = time.perf_counter()-start
                    if final and final.get('status') in ('completed', 'incomplete'):
                        usage = final.get('usage') or {}
                        actual = usage.get('input_tokens')
                        if type(actual) is not int or actual < args.prompt_tokens:
                            raise ValueError('Actual input token usage is missing or below the filler budget')
                    return {'request':index,'status':final['status'] if final else 'NO_TERMINAL',
                            'seconds':elapsed,'ttft_seconds':ttft,'usage':final.get('usage') if final else None,
                            'needle_found':marker in text,'output_excerpt':text[:512]}
                except Exception as exc:
                    return {'request':index,'status':'ERROR','error':f'{type(exc).__name__}: {exc}',
                            'seconds':time.perf_counter()-start}
        start = time.perf_counter()
        results = await asyncio.gather(*(one(i) for i in range(args.requests)))
        elapsed = time.perf_counter()-start
    total = sum((r.get('usage') or {}).get('output_tokens',0) for r in results)
    report = {'status':'MEASURED','compatibility':info,'concurrency':args.concurrency,
              'requested_prompt_tokens':args.prompt_tokens,'actual_filler_tokens':args.prompt_tokens,
              'prompt_budget_scope':'filler only; instructions and official template add tokens',
              'elapsed_seconds':elapsed,
              'output_tokens_per_second':total/elapsed,'requests':results,
              'all_requests_finished':all(r['status'] in ('completed','incomplete') for r in results),
              'all_needles_found':all(r.get('needle_found',False) for r in results)}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='requests'},indent=2))
    return 0 if report['all_requests_finished'] and report['all_needles_found'] else 1


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument('--url',default='http://127.0.0.1:8005')
    p.add_argument('--uds', default=None)
    p.add_argument('--concurrency',type=int,default=32)
    p.add_argument('--requests',type=int,default=32)
    p.add_argument('--prompt-tokens',type=int,default=8192)
    p.add_argument('--output-tokens',type=int,default=128)
    p.add_argument('--timeout',type=float,default=14400)
    p.add_argument('--tokenizer',type=Path,default=root/'vendor/tokenizer/tokenizer.json')
    p.add_argument('--report',type=Path,required=True)
    args = p.parse_args()
    if min(args.concurrency,args.requests,args.prompt_tokens,args.output_tokens) < 1:
        p.error('All sizes must be positive')
    raise SystemExit(asyncio.run(run(args)))
