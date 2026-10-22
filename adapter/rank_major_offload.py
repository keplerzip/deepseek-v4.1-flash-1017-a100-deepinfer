"""Contiguous per-rank CPU backing for the pinned native offloading handlers.

No attention kernels, cache keys, ARC policy or GPU layout changes.
In-memory only: all ranks in an engine use this version; no old-cache import.
"""
import math
import mmap
import os
import time

import torch
from vllm.logger import init_logger
from vllm.v1.kv_offload.cpu import gpu_worker as native_worker
from vllm.v1.kv_offload.cpu.shared_offload_region import (
    SharedOffloadRegion, _get_populate_write_fn,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec, _all_workers_barrier
from adapter.numa_policy import interleave_mapping

logger=init_logger(__name__)
PIN_BATCH_BYTES=1<<30


def rank_geometry(num_chunks,cpu_page_size,world_size,rank):
    if not (num_chunks>0 and cpu_page_size>0 and 0<=rank<world_size):
        raise ValueError('Invalid rank-major geometry')
    span=num_chunks*cpu_page_size
    if span % mmap.PAGESIZE:
        raise ValueError('Rank boundaries must be OS-page aligned')
    return rank*span,span


class RankMajorRegion(SharedOffloadRegion):
    def _map_and_populate(self,rank,num_chunks,cpu_page_size):
        if rank is None or self._row_stride % cpu_page_size:
            raise ValueError('Rank-major CPU cache requires exact worker slots')
        workers=self._row_stride//cpu_page_size
        offset,span=rank_geometry(num_chunks,cpu_page_size,workers,rank)
        if offset+span>self.total_size_bytes:raise ValueError('Rank-major mapping overflow')
        self._rank_major_base=offset
        self._rank_major_span=span
        self._row_stride=cpu_page_size
        self._worker_offset=offset
        self._worker_area_end=offset+cpu_page_size
        self.mmap_obj.madvise(getattr(mmap,'MADV_NOHUGEPAGE',15))
        policy=interleave_mapping(self.mmap_obj)
        logger.info('HF3 CPU KV populate start rank=%d GiB=%.3f NUMA=%s',rank,span/(1<<30),policy)
        started=time.monotonic()
        _get_populate_write_fn(self.mmap_obj)(self.mmap_obj,offset,span)
        self._base=torch.frombuffer(memoryview(self.mmap_obj),dtype=torch.int8)
        self._views=[]
        self._canonical_offset=0
        self.is_pinned=False
        self._pinned_slot_offsets=[]
        logger.info('HF3 CPU KV populate done rank=%d elapsed=%.1fs',rank,time.monotonic()-started)

    def create_next_canonical_view(self,tensor_page_size):
        raise RuntimeError('Canonical/topology-independent offload is outside this layout')

    def create_kv_memoryview(self):
        raise RuntimeError('Secondary tiers are not configured for this rank-major layout')


def _probe_device():
    probe=torch.zeros(8,device='cuda')
    probe+=1
    torch.cuda.synchronize()


def pin_rank_major(region,cudart=None):
    if os.environ.get('VLLM_KV_OFFLOAD_SKIP_PIN')=='1':
        raise RuntimeError('HF3 requires actual host-memory pinning')
    base=region._base.data_ptr()
    offset=region._rank_major_base
    span=region._rank_major_span
    if (base+offset)%mmap.PAGESIZE or span%mmap.PAGESIZE:
        raise ValueError('Unaligned host registration range')
    cudart=torch.cuda.cudart() if cudart is None else cudart
    registered=[]
    started=time.monotonic()
    batches=math.ceil(span/PIN_BATCH_BYTES)
    logger.info('HF3 pin start rank=%d GiB=%.3f registrations=%d',region.rank,span/(1<<30),batches)
    for done in range(0,span,PIN_BATCH_BYTES):
        position=offset+done
        size=min(PIN_BATCH_BYTES,span-done)
        for attempt in range(4):
            result=cudart.cudaHostRegister(base+position,size,0)
            if result.value==0:break
            logger.warning('HF3 pin retry rank=%d batch=%d/%d attempt=%d code=%d',
                           region.rank,len(registered)+1,batches,attempt+1,result.value)
            try:_probe_device()
            except Exception:pass
            time.sleep(.5*(attempt+1))
        if result.value:
            for previous in registered:cudart.cudaHostUnregister(base+previous)
            raise RuntimeError(f'HF3 cudaHostRegister rank={region.rank} batch={len(registered)+1}/{batches} code={result.value}')
        registered.append(position)
        logger.info('HF3 pin progress rank=%d batch=%d/%d GiB=%.3f elapsed=%.1fs',
                    region.rank,len(registered),batches,(done+size)/(1<<30),time.monotonic()-started)
    region._pinned_slot_offsets=registered
    region.is_pinned=True
    logger.info('HF3 pin done rank=%d GiB=%.3f registrations=%d elapsed=%.1fs',
                region.rank,span/(1<<30),len(registered),time.monotonic()-started)
    try:_probe_device()
    except Exception:
        logger.warning('HF3 clearing stale CUDA error after host registration; checking device again')
        _probe_device()


_original_pin=native_worker.pin_mmap_region

def _dispatch_pin(region):
    if isinstance(region,RankMajorRegion):return pin_rank_major(region)
    return _original_pin(region)

native_worker.pin_mmap_region=_dispatch_pin


class RankMajorCPUOffloadingSpec(CPUOffloadingSpec):
    def __init__(self,config):
        super().__init__(config)
        parallel=config.parallel
        if (config.canonical_layout or self.replicated_layout or
            parallel.tp_size!=8 or parallel.world_size!=8 or parallel.pp_size!=1 or
            parallel.pcp_size!=1 or parallel.dcp_size!=1 or parallel.data_parallel_size!=1):
            raise ValueError('HF3 rank-major layout is scoped to noncanonical TP8/PP1/DP1')
        multiple=mmap.PAGESIZE//math.gcd(mmap.PAGESIZE,self.cpu_page_size_per_worker)
        self.num_chunks-=self.num_chunks%multiple
        if self.num_chunks<=0:raise ValueError('CPU KV budget cannot hold an aligned rank extent')
        logger.info('HF3 rank-major CPU KV chunks=%d total_GiB=%.6f per_rank_GiB=%.6f registrations_per_rank=%d',
            self.num_chunks,self.num_chunks*self.kv_bytes_per_chunk/(1<<30),
            self.num_chunks*self.cpu_page_size_per_worker/(1<<30),
            math.ceil(self.num_chunks*self.cpu_page_size_per_worker/PIN_BATCH_BYTES))

    def create_worker(self,kv_caches):
        if not self._uses_shared_region():
            raise RuntimeError('HF3 requires the CUDA shared-memory backing')
        rank=torch.accelerator.current_device_index()%self.config.parallel.world_size
        region=RankMajorRegion(engine_id=self.config.engine_id,num_chunks=self.num_chunks,
            rank=rank,kv_bytes_per_chunk=self.kv_bytes_per_chunk,
            cpu_page_size=self.cpu_page_size_per_worker,barrier=_all_workers_barrier)
        try:
            return native_worker.CPUOffloadingWorker(kv_caches=kv_caches,
                blocks_per_chunk=self.blocks_per_chunk,num_cpu_chunks=self.num_chunks,mmap_region=region)
        except Exception:
            region.cleanup()
            raise
