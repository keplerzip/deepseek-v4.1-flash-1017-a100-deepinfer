"""Real-model CPU-tier proof: store, reset GPU-only cache, reload, compare output."""
import argparse
import json
from pathlib import Path
import re
import time
import uuid
from live_acceptance import request, output_text, function_schema


def metric_total(text, name):
    total=0.0
    for line in text.splitlines():
        if not line or line.startswith('#'):continue
        m=re.match(r'([^\s{]+)(?:\{[^}]*\})?\s+(\S+)',line)
        if m and m[1] in (name,name+'_total'):
            total+=float(m[2])
    return total


def metrics(sock):
    raw=request(sock,'GET','/metrics').decode()
    return {kind:metric_total(raw,'vllm:kv_offload_'+kind+'_bytes') for kind in ('load','store')}


def signature(response, use_tool):
    if response.get('status') != 'completed':raise ValueError('Real-model response did not complete')
    if not use_tool:return output_text(response)
    return [(item['name'],json.loads(item['arguments'])) for item in response['output'] if item['type']=='function_call']


def run(sock, use_tool=False):
    before=metrics(sock)
    marker='RAM_KV_'+uuid.uuid4().hex
    prompt=f'The exact secret is {marker}.\n'+('A stable reference paragraph for testing prefix cache reuse.\n'*2048)+f'\nReply with only {marker}.'
    body={'model':'DeepSeek-Flash','input':prompt,'reasoning':{'effort':'none'},
          'temperature':0,'max_output_tokens':64}
    if use_tool:
        body.update(tools=[function_schema()],tool_choice='required')
        body['input']=prompt.rsplit('\nReply with only ',1)[0]+f'\nCall echo_probe once with marker exactly {marker}.'
    first=json.loads(request(sock,'POST','/v1/responses',body))
    expected=signature(first,use_tool)
    if (use_tool and expected != [('echo_probe',{'marker':marker})]) or (not use_tool and marker not in expected):
        raise ValueError('First real-model response did not reproduce its text/tool marker')
    deadline=time.monotonic()+180
    stored=before
    while time.monotonic()<deadline:
        stored=metrics(sock)
        if stored['store']>before['store']:
            reset=json.loads(request(sock,'POST','/deepinfer/cache/reset-local'))
            if reset.get('success') is True:break
        time.sleep(1)
    else:raise ValueError('No CPU KV store completed, or GPU-only prefix reset remained busy')
    second=json.loads(request(sock,'POST','/v1/responses',body))
    after=metrics(sock)
    if signature(second,use_tool)!=expected:
        raise ValueError('Greedy output changed across CPU KV restore')
    for _ in range(30):
        if after['load']>stored['load']:break
        time.sleep(1)
        after=metrics(sock)
    else:raise ValueError('No bytes actually reloaded from CPU after clearing the GPU prefix cache')
    cached=second.get('usage',{}).get('input_tokens_details',{}).get('cached_tokens',0)
    if cached<=0:raise ValueError('Restored request reported no cached prompt tokens')
    return {'status':'PASS','mode':'tools' if use_tool else 'text','gpu_only_prefix_reset':True,'cpu_store_bytes':stored['store']-before['store'],
            'cpu_reload_bytes':after['load']-stored['load'],'cached_prompt_tokens':cached,
            'greedy_output_identical':True,'second_usage':second['usage'],
            'scope':'Actual model and CPU->GPU tier restore; not a throughput or statistical quality claim'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--socket',default='/state/api.sock')
    p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    plan=json.loads(Path('/state/cpu-kv-plan.json').read_text())
    if not plan['effective_gib']:
        report={'status':'SKIP','reason':'CPU KV explicitly disabled','cpu_tier_tested':False}
    else:
        try:report={'status':'PASS','cases':[run(a.socket),run(a.socket,use_tool=True)],'scope':'Real text and official tool-call CPU KV restore'}
        except Exception as exc:report={'status':'FAIL','error':f'{type(exc).__name__}: {exc}'}
    a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return int(report['status']=='FAIL')
if __name__=='__main__':raise SystemExit(main())
