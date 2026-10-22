"""Real pinned native-connector CPU contracts, not GPU/DMA evidence."""
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from vllm import SamplingParams
from vllm.config import CacheConfig
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole, supports_hma
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import build_offloading_config
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import OffloadingWorkerMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector as Native
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.output import CachedRequestData, KVConnectorBlockState, NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import CircularBufferSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, MLAAttentionSpec, SlidingWindowMLASpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus
from adapter.native_kv_offload import OffloadingConnector


def fixture(ring_index=1):
    config = NS(
        cache_config=CacheConfig(block_size=32, cache_dtype='fp8_ds_mla', enable_prefix_caching=True),
        kv_transfer_config=KVTransferConfig(kv_connector='OffloadingConnector', kv_role='kv_both',
            kv_connector_module_path='adapter.native_kv_offload', kv_connector_extra_config={
                'cpu_bytes_to_use': 16 << 20, 'offload_prompt_only':False,
                'blocks_per_chunk':1,'eviction_policy':'arc',
                'spec_name':'RankMajorCPUOffloadingSpec','spec_module_path':'adapter.rank_major_offload'}),
        parallel_config=NS(decode_context_parallel_size=1,prefill_context_parallel_size=1,
            tensor_parallel_size=8,pipeline_parallel_size=1,world_size=8,rank=0,
            distributed_executor_backend='mp',nnodes_within_dp=1,data_parallel_index=0,
            data_parallel_size=1,data_parallel_rank_local=0),
        model_config=NS(use_mla=True,dtype=torch.bfloat16,model='1017', get_total_num_kv_heads=lambda:1),
        scheduler_config=NS(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,kv_events_config=None,use_v2_model_runner=False)
    groups=[KVCacheGroupSpec(['mla'],MLAAttentionSpec(block_size=32,num_kv_heads=1,
                head_size=512,dtype=torch.uint8,tokens_per_state=2)),
            KVCacheGroupSpec(['swa'],SlidingWindowMLASpec(block_size=32,num_kv_heads=1,
                head_size=512,dtype=torch.uint8,sliding_window=128))]
    groups.insert(ring_index,KVCacheGroupSpec(['ring'],CircularBufferSpec(block_size=8,
                num_kv_heads=1,head_size=1024,head_size_v=0,dtype=torch.float32)))
    cache=KVCacheConfig(num_blocks=16,kv_cache_tensors=[KVCacheTensor(size=16*65536,
                layers=['mla','swa','ring'],layer_stride=16384,block_stride=65536)],kv_cache_groups=groups)
    return config,cache


class NativeKVOffloadTests(unittest.TestCase):
    def connector(self, ring_index=1, role=KVConnectorRole.SCHEDULER):
        config,cache=fixture(ring_index)
        return OffloadingConnector(config,role,cache),config,cache

    def test_unpatched_native_reproduces_target_8_vs_32_assertion(self):
        config,cache=fixture()
        with self.assertRaisesRegex(AssertionError,'tokens_per_block=8 not divisible by tokens_per_hash=32'):
            build_offloading_config(config,cache)

    def test_real_native_scheduler_and_worker_initialize_with_ring_in_any_position(self):
        for index in range(3):
            for role in [KVConnectorRole.SCHEDULER,KVConnectorRole.WORKER]:
                connector,config,original=self.connector(index,role)
                self.assertEqual(len(original.kv_cache_groups),3)
                self.assertEqual(len(connector._kv_cache_config.kv_cache_groups),2)
                translated=build_offloading_config(config,connector._kv_cache_config)
                self.assertEqual(translated.cache.tokens_per_hash,32)
                self.assertEqual([g.tokens_per_block for g in translated.groups],[32,32])
                self.assertEqual(translated.extra_config['eviction_policy'],'arc')
                if role==KVConnectorRole.SCHEDULER:
                    self.assertEqual([g.group_idx for g in connector.connector_scheduler.config.kv_group_configs],[0,1])
                connector.shutdown()

    def test_native_factory_loads_wrapper_and_keeps_hma(self):
        config,cache=fixture()
        self.assertTrue(KVConnectorFactory.supports_hma_config(config.kv_transfer_config))
        connector=KVConnectorFactory.create_connector(config,KVConnectorRole.SCHEDULER,cache)
        self.assertIsInstance(connector,Native)
        self.assertTrue(supports_hma(type(connector)))
        self.assertFalse(connector.requires_kv_delivery)

    def test_block_allocation_projection_preserves_scratch_overwrite_fence(self):
        connector,_,_=self.connector()
        blocks=KVCacheBlocks(tuple([KVCacheBlock(block_id=i)] for i in [11,22,33]))
        with patch.object(Native,'update_state_after_alloc') as native:
            connector.update_state_after_alloc('request',blocks,64)
        self.assertEqual(native.call_args.args[1].get_block_ids(),([11],[33]))
        self.assertEqual(blocks.get_block_ids(),([11],[22],[33]))
        self.assertEqual(connector.connector_scheduler._current_batch_allocated_block_ids,{11,22,33})

    def test_new_cached_resumed_requests_and_boundary_handoffs_project_without_mutation(self):
        connector,_,_=self.connector()
        original=([11],[22],[33])
        new=NewRequestData('new',[1],[],None,None,original,0,None)
        cached=CachedRequestData(req_ids=['cached','unchanged','resumed'],resumed_req_ids={'resumed'},
            new_token_ids=[[],[],[]],all_token_ids={},new_block_ids=[original,None,original],
            num_computed_tokens=[32,32,32],num_output_tokens=[0,0,0])
        state=KVConnectorBlockState({'new'},lambda req:original,{'new':[(0,11,32),(1,22,32),(2,33,32)]})
        output=replace(SchedulerOutput.make_empty(),scheduled_new_reqs=[new],scheduled_cached_reqs=cached,
                       kv_connector_block_state=state)
        with patch.object(Native,'build_connector_meta',return_value='native-meta') as native:
            self.assertEqual(connector.build_connector_meta(output),'native-meta')
        projected=native.call_args.args[0]
        self.assertEqual(projected.scheduled_new_reqs[0].block_ids,([11],[33]))
        self.assertEqual(projected.scheduled_cached_reqs.new_block_ids,[([11],[33]),None,([11],[33])])
        self.assertEqual(projected.kv_connector_block_state.get_block_ids('new'),([11],[33]))
        self.assertEqual(projected.kv_connector_block_state.boundary_state_offloads,{'new':[(0,11,32),(1,33,32)]})
        self.assertEqual(output.scheduled_new_reqs[0].block_ids,original)
        self.assertEqual(output.scheduled_cached_reqs.new_block_ids,[original,None,original])
        self.assertEqual(state.boundary_state_offloads['new'],[(0,11,32),(1,22,32),(2,33,32)])
        self.assertEqual(connector.connector_scheduler._current_batch_allocated_block_ids,{11,22,33})

    def test_worker_registration_excludes_ring_only(self):
        connector,_,_=self.connector(role=KVConnectorRole.WORKER)
        with patch.object(Native,'register_kv_caches') as native:
            connector.register_kv_caches({'mla':'M','ring':'R','swa':'S'})
        self.assertEqual(native.call_args.args[0],{'mla':'M','swa':'S'})

    def test_bad_group_count_fails_and_empty_or_none_blocks_survive(self):
        connector,_,_=self.connector()
        with self.assertRaisesRegex(ValueError,'group count'):connector._select(([1],))
        self.assertIsNone(connector._select(None))
        self.assertEqual(connector._select(()),())

    def test_ring_reuse_flushes_pending_native_store_before_overwrite(self):
        connector,_,_=self.connector()
        scheduler=connector.connector_scheduler
        scheduler._block_id_to_pending_jobs[22]={91}
        output=replace(SchedulerOutput.make_empty(),scheduled_cached_reqs=CachedRequestData(
            req_ids=['cached'],resumed_req_ids=set(),new_token_ids=[[]],all_token_ids={},
            new_block_ids=[([11],[22],[33])],num_computed_tokens=[32],num_output_tokens=[0]))
        with patch.object(scheduler,'_update_req_states'),patch.object(scheduler,'_build_store_jobs',return_value={}):
            meta=connector.build_connector_meta(output)
        self.assertIn(91,meta.jobs_to_flush)
        self.assertEqual(scheduler._current_batch_allocated_block_ids,set())

    def test_native_store_completion_lookup_and_load_metadata_exclude_ring(self):
        connector,_,_=self.connector()
        def request(name):
            req=Request(name,list(range(96)),SamplingParams(max_tokens=8),None)
            req.block_hashes=[bytes([i])*32 for i in range(1,4)]
            connector.on_new_request(req)
            return req
        first=request('first')
        self.assertEqual(connector.get_num_new_matched_tokens(first,0),(0,False))
        first.status=RequestStatus.RUNNING
        original=([1,2,3],[7],[4,5,6])
        output=replace(SchedulerOutput.make_empty(),
            scheduled_new_reqs=[NewRequestData.from_request(first,original)],
            num_scheduled_tokens={'first':96},total_num_scheduled_tokens=96)
        stores=connector.build_connector_meta(output)
        self.assertTrue(stores.store_jobs)
        for job in stores.store_jobs.values():
            self.assertEqual(len(job.src_spec.group_sizes),2)
            self.assertNotIn(7,job.src_spec.block_ids)
        completion=OffloadingWorkerMetadata()
        completion.completed_jobs={job:8 for job in stores.store_jobs}
        connector.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=completion))
        second=request('second')
        hit,asynchronous=connector.get_num_new_matched_tokens(second,0)
        self.assertEqual(hit,64)
        self.assertTrue(asynchronous)
        blocks=KVCacheBlocks(tuple([KVCacheBlock(block_id=i) for i in group]
                                  for group in ([8,9],[12],[10,11])))
        connector.update_state_after_alloc(second,blocks,hit)
        loads=connector.build_connector_meta(SchedulerOutput.make_empty())
        self.assertTrue(loads.load_jobs)
        for job in loads.load_jobs.values():
            self.assertEqual(job.dst_spec.group_sizes,[2,2])
            self.assertEqual(list(job.dst_spec.block_ids),[8,9,10,11])
        self.assertEqual(output.scheduled_new_reqs[0].block_ids,original)

    def test_native_worker_registers_actual_packed_cpu_views_with_two_transfer_groups(self):
        connector,_,_=self.connector(role=KVConnectorRole.WORKER)
        storage=torch.empty(16*65536,dtype=torch.uint8)
        mla=torch.as_strided(storage,(16,1,16,512),(65536,8192,512,1),0)
        swa=torch.as_strided(storage,(16,1,32,512),(65536,16384,512,1),8192)
        worker=connector.connector_worker
        with patch('vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker.derive_canonical_mappings',return_value={}),patch.object(worker,'_init_worker') as initialize:
            connector.register_kv_caches({'mla':mla,'swa':swa,'ring':object()})
        registered=initialize.call_args.args[0]
        self.assertEqual(len(registered.group_data_refs),2)
        self.assertEqual(len(registered.tensors),1)
        self.assertEqual(registered.tensors[0].tensor.shape,(16,65536))

if __name__=='__main__':unittest.main()
