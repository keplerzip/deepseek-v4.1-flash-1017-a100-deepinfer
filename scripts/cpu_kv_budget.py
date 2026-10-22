"""Budget native CPU KV globally across TP ranks and guard host RAM reserve."""
import argparse
import json
import os
from pathlib import Path

GIB = 1 << 30
ROOT = Path(__file__).resolve().parents[1]


def available_bytes(meminfo=Path('/proc/meminfo'), cgroup=Path('/sys/fs/cgroup')):
    fields = {k: int(v.split()[0])*1024 for k,v in
              (line.split(':',1) for line in meminfo.read_text().splitlines() if ':' in line)}
    available = fields['MemAvailable']
    for limit, usage in [('memory.max','memory.current'),
                         ('memory/memory.limit_in_bytes','memory/memory.usage_in_bytes')]:
        lp,up=cgroup/limit,cgroup/usage
        if lp.is_file() and up.is_file() and lp.read_text().strip() != 'max':
            available=min(available,max(0,int(lp.read_text())-int(up.read_text())))
    return available


def plan(settings, available, shm_available, requested=None, policy=None):
    target=settings['target_gib'] if requested is None else requested
    if type(target) is not int or target < 0 or target > settings['target_gib']:
        raise ValueError('CPU KV target must be 0..768 GiB TOTAL across TP8')
    policy=settings.get('eviction_policy','arc') if policy is None else policy
    if policy not in ('arc','lru'):raise ValueError('CPU KV policy must be arc or lru')
    reserve=settings['reserve_available_gib']
    fixed=reserve+settings['engine_startup_headroom_gib']+settings['allocation_overhead_gib']
    limit=min(target, max(0, available//GIB-fixed),
              max(0,shm_available//GIB-settings['shm_headroom_gib']))
    chosen=(limit//settings['rounding_gib'])*settings['rounding_gib']
    if target and chosen < settings['rounding_gib']:
        raise ValueError('Insufficient available RAM or /dev/shm while retaining 500 GiB and engine startup headroom')
    if available < reserve*GIB:
        raise ValueError('Host available memory is already below the 500 GiB reserve')
    return {'status':'PASS','scope':'TOTAL CPU KV capacity across TP8, not per GPU',
            'requested_gib':target,'effective_gib':chosen,'host_available_bytes':available,
            'shm_available_bytes':shm_available,'reserve_available_gib':reserve,
            'engine_startup_headroom_gib':settings['engine_startup_headroom_gib'],
            'allocation_overhead_gib':settings['allocation_overhead_gib'],
            'eviction_policy':policy,'offload_prompt_only':False,
            'persistence':'RAM only; lost on engine restart',
            'actual_reserve_after_start':'NOT_YET_CHECKED'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--requested-gib',type=int)
    p.add_argument('--verify-reserve',action='store_true')
    p.add_argument('--policy', choices=('arc','lru'))
    a=p.parse_args()
    settings=json.loads((ROOT/'configs/cpu-kv.json').read_text())
    available=available_bytes()
    if a.verify_reserve:
        report={'status':'PASS' if available >= settings['reserve_available_gib']*GIB else 'FAIL',
                'host_available_bytes':available,'reserve_available_gib':settings['reserve_available_gib']}
    else:
        fs=os.statvfs('/dev/shm')
        report=plan(settings,available,fs.f_bavail*fs.f_frsize,a.requested_gib,a.policy)
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    return 0 if report['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
