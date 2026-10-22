"""Compare bundled native replacement policies on CPU; no GPU speed claim."""
import argparse
import json
from pathlib import Path
from vllm.v1.kv_offload.cpu.policies.factory import CachePolicyFactory
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus


def simulate(name, trace, capacity=16):
    policy=CachePolicyFactory.get_cache_policy_cls(name)(capacity)
    live=0;hits=0
    for value in trace:
        key=str(value).encode()
        chunk=policy.get(key)
        policy.touch([key],None)
        if chunk is not None:
            hits+=1
            continue
        if live==capacity:
            evicted=policy.evict(1,set())
            assert evicted is not None and len(evicted)==1
            chunk=evicted[0][1]
        else:
            chunk=ChunkStatus(live);chunk.ref_cnt=0;live+=1
        policy.insert(key,chunk)
    return {'accesses':len(trace),'hits':hits,'hit_ratio':hits/len(trace)}


def validate_safety(name):
    p=CachePolicyFactory.get_cache_policy_cls(name)(4)
    chunks=[]
    for i in range(4):
        c=ChunkStatus(i);c.ref_cnt=0;p.insert(bytes([i]),c);chunks.append(c)
    chunks[0].ref_cnt=1;p.mark_non_evictable(bytes([0]))
    assert p.evict(3,{bytes([1])}) is None
    assert all(p.get(bytes([i])) is chunks[i] for i in range(4))
    result=p.evict(2,{bytes([1])})
    assert result is not None and {key for key,_ in result}=={bytes([2]),bytes([3])}
    assert p.get(bytes([0])) is chunks[0] and p.get(bytes([1])) is chunks[1]
    p.clear();assert all(p.get(bytes([i])) is None for i in range(4))


def compare():
    hot=list(range(4));scan=hot*8
    for iteration in range(40):scan+=list(range(100+iteration*32,132+iteration*32))+hot*4
    stable=list(range(8))*100
    results={name:{'scan_with_hot_dialogues':simulate(name,scan),'stable_working_set':simulate(name,stable)}
             for name in ('lru','arc')}
    for name in ('lru','arc'):validate_safety(name)
    assert results['arc']['scan_with_hot_dialogues']['hits']>results['lru']['scan_with_hot_dialogues']['hits']
    assert results['arc']['stable_working_set']['hits']==results['lru']['stable_working_set']['hits']
    return {'status':'PASS','scope':'CPU replacement-policy trace and active-transfer protection only; no GPU or real workload performance claim',
            'capacity_chunks':16,'policies':results,'transfer_protection_and_atomic_failed_eviction':'PASS',
            'selected_default':'arc','lru_control_available':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    report=compare();a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
