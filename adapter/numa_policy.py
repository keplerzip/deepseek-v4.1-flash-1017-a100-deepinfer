"""NUMA policy for this offload mapping only; no host-wide settings."""
import ctypes
import json
import mmap
from pathlib import Path
import platform


def allowed_nodes(status=Path('/proc/self/status')):
    raw=next(line.split(':',1)[1].strip() for line in status.read_text().splitlines()
             if line.startswith('Mems_allowed_list:'))
    result=[]
    for part in raw.split(','):
        ends=list(map(int,part.split('-')))
        result.extend(range(ends[0],ends[-1]+1))
    if not result:raise RuntimeError('No allowed NUMA memory nodes')
    return result


def interleave_mapping(mapping):
    nodes=allowed_nodes()
    if platform.machine() not in ('x86_64','AMD64'):
        raise RuntimeError('This pinned NUMA syscall binding requires x86_64')
    words=(max(nodes)+64)//64
    mask=(ctypes.c_ulong*words)()
    for node in nodes:mask[node//64] |= 1 << (node%64)
    buffer=ctypes.c_char.from_buffer(mapping)
    try:
        libc=ctypes.CDLL(None,use_errno=True)
        result=libc.syscall(237,ctypes.c_void_p(ctypes.addressof(buffer)),
            ctypes.c_size_t(len(mapping)),ctypes.c_int(3),mask,
            ctypes.c_ulong(max(nodes)+2),ctypes.c_uint(0))
        if result:
            code=ctypes.get_errno()
            raise OSError(code,'mbind interleave denied; verify SYS_NICE/seccomp for this container')
    finally:del buffer
    return {'status':'PASS','numa_policy':'interleave','nodes':nodes}


if __name__=='__main__':
    with mmap.mmap(-1,2<<20) as mapping:
        print(json.dumps(interleave_mapping(mapping)),flush=True)
