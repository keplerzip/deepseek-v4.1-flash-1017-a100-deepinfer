#!/usr/bin/env python3
"""Paired real-model quality checks. Never executes generated code or tools."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import random
import re
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = '1017-r8-quality-v1'


def request(base, path, body=None, timeout=1800):
    raw = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(base.rstrip('/') + path, data=raw,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def number(text):
    matches = re.findall(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)', text)
    if not matches:
        return None
    try:
        return str(Decimal(matches[-1].replace(',', '')).normalize())
    except InvalidOperation:
        return None


def body_for(prompt, mode, budget):
    return {'model': 'DeepSeek-Flash', 'messages': [{'role': 'user', 'content': prompt}],
        'thinking': {'type': 'disabled' if mode == 'off' else 'enabled'},
        'reasoning_effort': 'none' if mode == 'off' else 'high',
        'temperature': 0, 'top_p': 1, 'seed': 1017, 'max_tokens': budget,
        'stream': False}


def evaluate(base, case):
    start = time.monotonic()
    row = {k: case[k] for k in ('id', 'kind', 'mode')}
    row['request_sha256'] = hashlib.sha256(json.dumps(case['body'], sort_keys=True,
        ensure_ascii=False).encode()).hexdigest()
    try:
        result = request(base, '/chat/completions', case['body'])
        choice = result['choices'][0]
        text = choice['message'].get('content') or ''
        usage = result.get('usage', {})
        complete = choice.get('finish_reason') == 'stop'
        valid = not re.search(r'<[｜|]DSML[｜|]>|</parameter>|</invoke>|</function_calls>', text)
        if case['kind'] == 'gsm8k':
            correct = number(text) == case['answer']
        elif case['kind'] == 'long':
            try:
                clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
                correct = json.loads(clean) == case['answer']
            except (ValueError, TypeError):
                correct = False
            count = usage.get('prompt_tokens')
            if count != case['prompt_tokens']:
                raise ValueError('Server/local official prompt token counts differ')
            if count < case['target_tokens'] - 128:
                raise ValueError('Long-context request did not reach its declared length')
        else:
            correct = text.strip() == case['answer']
        row.update(status='PASS' if correct and complete and valid else 'FAIL',
            correct=correct, completed=complete, protocol_clean=valid,
            finish_reason=choice.get('finish_reason'), usage=usage,
            answer_sha256=hashlib.sha256(text.encode()).hexdigest(),
            # Synthetic/public prompts only; useful for reviewing paired misses.
            answer=text, expected=case['answer'])
    except Exception as exc:
        row.update(status='ERROR', error=type(exc).__name__ + ': ' + str(exc))
    row['seconds'] = time.monotonic() - start
    return row


def gsm_cases(limit):
    data = [json.loads(x) for x in (ROOT/'vendor/eval/gsm8k-test.jsonl').read_text().splitlines()]
    indices = list(range(len(data)))
    random.Random(1017).shuffle(indices)
    for mode in ('off', 'high'):
        for i in sorted(indices[:limit]):
            item = data[i]
            prompt = item['question'] + '\nSolve carefully. End with #### followed by the numeric answer.'
            yield dict(id=f'gsm8k/{mode}/{i:04}', kind='gsm8k', mode=mode,
                body=body_for(prompt, mode, 2048 if mode == 'off' else 4096),
                answer=number(item['answer']))


def isolation_cases():
    # Deterministic, different request-local facts, sent concurrently at C32.
    for i in range(64):
        value = hashlib.sha256(f'1017-isolation-{i}'.encode()).hexdigest()[:24]
        yield dict(id=f'isolation/off/{i:03}', kind='isolation', mode='off',
            body=body_for(f'For this request only, the secret marker is {value}. '
                          'Reply with only that marker, without quotes.', 'off', 64), answer=value)


def code_cases():
    for mode in ('off', 'high'):
        for i in range(32):
            a,b,m=3+i,7+i,11+i
            value=a
            for j in range(1,8):
                value=(value*j+b)%m
            code=f"x = {a}\nfor j in range(1, 8):\n    x = (x * j + {b}) % {m}\nprint(x)"
            yield dict(id=f'code/{mode}/{i:03}',kind='code',mode=mode,
                body=body_for('Determine the exact output of this Python program. '
                    'Reply with the output only, without formatting.\n'+code,mode,2048 if mode=='off' else 4096),
                answer=str(value))


def prompt_counter():
    import deepseek_recipe as recipe
    from adapter.official_server import load_official, options, SPECIAL
    official = load_official()
    tok = recipe.Tokenizer.from_file(str(ROOT/'vendor/tokenizer/tokenizer.json'))
    enc = recipe.DeepseekV41Encoding().with_special_tokens(SPECIAL).with_tokenizer(tok)
    def count(body):
        converted = official.prepare_request('chat_completions', json.dumps(body).encode(), options=options())
        return len(enc.encode(converted.conversation_request.conversation))
    return count


def long_case(target, count):
    rng = random.Random(1017 + target)
    lines = [f'row {i:06d}: unused record {rng.getrandbits(64):016x}\n' for i in range(target // 8 + 1024)]
    keys = ['probe_' + hashlib.sha256(f'{target}-{i}'.encode()).hexdigest()[:12] for i in range(3)]
    values = [hashlib.sha256(f'value-{target}-{i}'.encode()).hexdigest()[:20] for i in range(3)]
    expected = dict(zip(keys, values))
    def make(n):
        selected = list(lines[:n])
        for f, key, value in zip((.05, .5, .95), keys, values):
            selected[min(n-1, int(n*f))] = f'Important record: key={key}; value={value}\n'
        prompt = 'Read these records and retrieve the requested values.\n' + ''.join(selected)
        prompt += '\nReturn only a JSON object mapping these keys to their values: ' + ', '.join(keys)
        return body_for(prompt, 'off', 256)
    lo, hi = 4, len(lines)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(make(mid)) <= target:
            lo = mid
        else:
            hi = mid - 1
    body = make(lo)
    actual = count(body)
    if not target - 128 <= actual <= target:
        raise ValueError(f'Cannot construct declared long prompt: {actual} vs {target}')
    return dict(id=f'long/off/{target}', kind='long', mode='off', body=body,
        answer=expected, prompt_tokens=actual, target_tokens=target)


def atomic_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def run(args):
    manifest = json.loads((ROOT/'vendor/eval/SOURCE.json').read_text())
    report = dict(schema=SCHEMA, label=args.label, status='RUNNING', real_model_inference=True,
        started_at=datetime.now(timezone.utc).isoformat(), endpoint=args.base_url,
        dataset_sha256=manifest['test_sha256'], gsm_limit=args.gsm_limit,
        sampling={'temperature':0, 'top_p':1, 'seed':1017}, cases=[])
    report['compatibility'] = request(args.base_url.removesuffix('/v1'), '/deepinfer/compatibility')
    atomic_write(args.output, report)
    cases = list(gsm_cases(args.gsm_limit)) + list(code_cases()) + list(isolation_cases())
    # GSM at C8, then a separate C32 isolation phase, avoiding huge prefill
    # sharing a decode step with the answer-quality sample phase.
    for kind, concurrency in [('gsm8k',8), ('code',8), ('isolation',32)]:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(evaluate,args.base_url,c) for c in cases if c['kind']==kind]
            for future in as_completed(futures):
                row=future.result();report['cases'].append(row);atomic_write(args.output,report)
                print(json.dumps({k:row[k] for k in ('id','status','seconds')}),flush=True)
    if args.long_context:
        count=prompt_counter()
        for target in (32768,131072,262144,523776):
            row=evaluate(args.base_url,long_case(target,count));report['cases'].append(row)
            atomic_write(args.output,report);print(json.dumps({k:row[k] for k in ('id','status','seconds')}),flush=True)
    report['cases'].sort(key=lambda r:r['id'])
    report['finished_at']=datetime.now(timezone.utc).isoformat()
    errors=[r for r in report['cases'] if r['status']=='ERROR' or r.get('completed') is False or r.get('protocol_clean') is False]
    report['status']='FAIL' if errors else 'MEASURED'
    report['long_context_512k']='PASS' if any(r['id']=='long/off/523776' and r['status']=='PASS' for r in report['cases']) else 'NOT_PASSED'
    report['scope']='Quality sample results, not a claim of no degradation; compare paired R7/R8 reports.'
    atomic_write(args.output,report)
    return 1 if errors else 0


def compare(before, after):
    a,b=(json.loads(Path(p).read_text()) for p in (before,after))
    if any(a.get(k)!=b.get(k) for k in ('schema','dataset_sha256','gsm_limit','sampling')):
        raise ValueError('Baseline/candidate test definitions differ')
    aa,bb=({r['id']:r for r in t['cases']} for t in (a,b))
    if aa.keys()!=bb.keys() or any(aa[k]['request_sha256']!=bb[k]['request_sha256'] for k in aa):
        raise ValueError('Baseline/candidate requests differ or reports are incomplete')
    groups={}
    regressions=[]
    for k in aa:
        x,y=aa[k],bb[k];group=x['kind']+'/'+x['mode']
        g=groups.setdefault(group,dict(samples=0,r7_pass=0,r8_pass=0,regressed=0,improved=0))
        old,new=x['status']=='PASS',y['status']=='PASS'
        g['samples']+=1;g['r7_pass']+=old;g['r8_pass']+=new
        g['regressed']+=old and not new;g['improved']+=new and not old
        if old and not new:regressions.append(k)
    issues=[]
    if len(aa)!=2*a['gsm_limit']+132:issues.append('Full long-context suite missing')
    if a['status']!='MEASURED' or b['status']!='MEASURED':issues.append('Protocol/request errors or incomplete run')
    if b.get('long_context_512k')!='PASS':issues.append('512K retrieval not passed')
    for group,g in groups.items():
        if group.startswith(('long/','isolation/')) and g['r8_pass']!=g['samples']:
            issues.append(group+' has failures')
        if group.startswith(('gsm8k/','code/')) and g['r8_pass']<g['r7_pass']:
            issues.append(group+' scored below R7')
    return dict(schema=SCHEMA,status='REVIEW_REQUIRED' if issues else 'PASS',groups=groups,
        paired_regressions=regressions,issues=issues,
        limitation='Finite deterministic paired sample only. Tool-probe and vision reports must also pass; no broad intelligence equivalence claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',default='http://127.0.0.1:8005/v1')
    p.add_argument('--label',default='R8');p.add_argument('--gsm-limit',type=int,default=200)
    p.add_argument('--long-context',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compare',nargs=2,metavar=('R7_REPORT','R8_REPORT'))
    args=p.parse_args()
    if args.compare:
        report=compare(*args.compare);atomic_write(args.output,report);print(json.dumps(report,indent=2));return 0 if report['status']=='PASS' else 1
    if not 1<=args.gsm_limit<=1319:p.error('gsm-limit must be 1..1319')
    return run(args)

if __name__=='__main__':raise SystemExit(main())
