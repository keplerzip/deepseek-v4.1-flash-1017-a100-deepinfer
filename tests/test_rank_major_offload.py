"""CPU byte-layout and simulated CUDART contracts; real GPU DMA is not run."""
from dataclasses import replace
import math
import gc
import mmap
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import uuid

import torch
from vllm.v1.kv_offload.base import CanonicalKVCaches,CanonicalKVCacheTensor,CanonicalKVCacheRef
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import build_offloading_config
from adapter import rank_major_offload as rank
from adapter.numa_policy import allowed_nodes
from test_native_kv_offload import fixture


class Runtime:
    def __init__(self,fail_after=None):self.calls=[];self.released=[];self.fail_after=fail_after
    def cudaHostRegister(self,address,size,flags):
        code=2 if self.fail_after is not None and len(self.calls)>=self.fail_after else 0
        self.calls.append((address,size,flags,code));return NS(value=code)
    def cudaHostUnregister(self,address):self.released.append(address);return NS(value=0)


def virtual_region(index=0,chunks=447392,slot=230400):
    offset,span=rank.rank_geometry(chunks,slot,8,index)
    return NS(rank=index,_base=NS(data_ptr=lambda:0x100000),_rank_major_base=offset,
              _rank_major_span=span,_pinned_slot_offsets=[],is_pinned=False)


class RankMajorTests(unittest.TestCase):
    def test_exact_reported_768gib_geometry_registers_96_batches_per_rank_without_overlap(self):
        intervals=[]
        with patch.object(rank,'_probe_device'):
            for index in range(8):
                runtime=Runtime();region=virtual_region(index)
                rank.pin_rank_major(region,runtime)
                self.assertEqual(len(runtime.calls),96)
                self.assertTrue(region.is_pinned)
                self.assertEqual(sum(size for _,size,_,_ in runtime.calls),region._rank_major_span)
                intervals.extend((address,address+size) for address,size,_,_ in runtime.calls)
                self.assertTrue(all(address%mmap.PAGESIZE==size%mmap.PAGESIZE==0 for address,size,_,_ in runtime.calls))
        intervals.sort()
        self.assertTrue(all(left[1]==right[0] for left,right in zip(intervals,intervals[1:])))
        self.assertEqual(intervals[-1][1]-intervals[0][0],447392*230400*8)

    def test_registration_failure_releases_successful_batches_and_never_marks_pinned(self):
        region=virtual_region();runtime=Runtime(fail_after=1)
        with patch.object(rank,'_probe_device'),patch.object(rank.time,'sleep'):
            with self.assertRaisesRegex(RuntimeError,'batch=2/96 code=2'):
                rank.pin_rank_major(region,runtime)
        self.assertFalse(region.is_pinned)
        self.assertEqual(len(runtime.calls),5)
        self.assertEqual(runtime.released,[0x100000])
        self.assertEqual(region._pinned_slot_offsets,[])

    def test_native_other_region_dispatch_is_preserved(self):
        other=object()
        with patch.object(rank,'_original_pin',return_value='original') as original:
            self.assertEqual(rank._dispatch_pin(other),'original')
            original.assert_called_once_with(other)

    def test_chunk_count_rounding_is_identical_for_manager_and_page_aligned_worker_extents(self):
        config,cache=fixture()
        cache=replace(cache,kv_cache_groups=[g for g in cache.kv_cache_groups if g.kv_cache_spec.prefix_cacheable])
        native=build_offloading_config(config,cache)
        for gib in [512,768]:
            settings=replace(native,worker_kv_bytes_per_block=230400,
                extra_config=dict(native.extra_config,cpu_bytes_to_use=gib*(1<<30)))
            spec=rank.RankMajorCPUOffloadingSpec(settings)
            self.assertEqual(spec.num_chunks%4,0)
            self.assertLessEqual(spec.num_chunks*spec.kv_bytes_per_chunk,gib*(1<<30))
            self.assertLess(gib*(1<<30)-spec.num_chunks*spec.kv_bytes_per_chunk,4*spec.kv_bytes_per_chunk)
            for index in range(8):
                base,span=rank.rank_geometry(spec.num_chunks,spec.cpu_page_size_per_worker,8,index)
                self.assertEqual(base%mmap.PAGESIZE,0)
                self.assertEqual(span%mmap.PAGESIZE,0)
            if gib==768:self.assertEqual(spec.num_chunks,447392)

    def test_actual_shared_views_are_byte_exact_across_all_ranks_and_native_worker_uses_new_strides(self):
        regions=[];views=[]
        first=second=raw=None
        engine='hf3-cpu-'+uuid.uuid4().hex
        with patch.object(rank,'interleave_mapping',return_value={'status':'CPU_TEST_POLICY'}):
            try:
                for index in range(8):
                    region=rank.RankMajorRegion(engine_id=engine,num_chunks=4,rank=index,
                        kv_bytes_per_chunk=5120*8,cpu_page_size=5120,barrier=None)
                    regions.append(region)
                    first=region.create_next_worker_view(768);second=region.create_next_worker_view(4352)
                    first.fill_(index+1);second.fill_(80+index);views.append((first,second))
                    self.assertEqual(first.stride(),(5120,1))
                    self.assertEqual(second.stride(),(5120,1))
                for index,(first,second) in enumerate(views):
                    offset,span=rank.rank_geometry(4,5120,8,index)
                    raw=regions[0]._base[offset:offset+span].view(4,5120)
                    self.assertTrue(torch.equal(raw[:,:768],first))
                    self.assertTrue(torch.equal(raw[:,768:],second))
                # Exercise original CPUOffloadingWorker assembly with a real
                # rank-major mmap; only GPU CUDART and DMA handlers are mocked.
                region=regions[-1];region._worker_offset=region._rank_major_base
                gpu_tensor=torch.arange(8*5120,dtype=torch.int32).to(torch.int8).view(8,5120)
                caches=CanonicalKVCaches([CanonicalKVCacheTensor(gpu_tensor,5120)],[[CanonicalKVCacheRef(0,5120)]])
                with patch.object(rank,'_probe_device'),patch.object(torch.cuda,'cudart',return_value=Runtime()),patch.object(rank.native_worker,'SingleDirectionOffloadingHandler') as handler:
                    worker=rank.native_worker.CPUOffloadingWorker(caches,1,4,mmap_region=region)
                self.assertEqual(handler.call_count,2)
                for call in handler.call_args_list:
                    cpu=call.kwargs['cpu_tensors'][0]
                    self.assertEqual(cpu.shape,(4,5120))
                    self.assertEqual(cpu.stride(),(5120,1))
                    self.assertEqual(cpu.data_ptr(),region._base.data_ptr()+region._rank_major_base)
                del worker,cpu,call,handler
            finally:
                views.clear()
                first=second=raw=None
                gc.collect()
                for region in regions:region.cleanup()

    def test_invalid_geometry_and_unsupported_secondary_tiers_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'OS-page aligned'):rank.rank_geometry(1,5120,8,0)
        region=rank.RankMajorRegion.__new__(rank.RankMajorRegion)
        with self.assertRaisesRegex(RuntimeError,'Secondary tiers'):region.create_kv_memoryview()
        with self.assertRaisesRegex(RuntimeError,'Canonical'):region.create_next_canonical_view(1)

    def test_proc_status_numa_node_ranges_are_respected(self):
        with patch('pathlib.Path.read_text',return_value='Name: test\nMems_allowed_list:\t0-2,5\n'):
            self.assertEqual(allowed_nodes(),[0,1,2,5])

if __name__=='__main__':unittest.main()
