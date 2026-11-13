"""Real metadata constructors with CPU buffers and simulated device capabilities."""
from contextlib import ExitStack
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
import torch
from vllm.config import AttentionConfig
from vllm.platforms.interface import DeviceCapability
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.attention.backends.mla import indexer as stock
from vllm.models.deepseek_v41.nvidia.specialized_decode import ampere
from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere.indexer_metadata import AmpereIndexerMetadataBuilder

class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.platform=NS(is_cuda=lambda:True,is_rocm=lambda:False,
            get_device_capability=lambda:DeviceCapability(8,0),
            is_device_capability_family=lambda f:f==80)
        self.ctx=ExitStack(); self.addCleanup(self.ctx.close)
        self.ctx.enter_context(patch.object(stock,'current_platform',self.platform))
        self.ctx.enter_context(patch.object(ampere,'current_platform',self.platform))
        self.ctx.enter_context(patch.object(stock,'has_deep_gemm',return_value=False))
        self.ctx.enter_context(patch.object(stock,'num_compute_units',return_value=108))
        self.cfg=NS(attention_config=AttentionConfig(indexer_kv_dtype='mxfp4',sparse_mla_prefill_workspace_factor=5),
            model_config=NS(max_model_len=524288,hf_text_config=NS(index_n_heads=32,index_head_dim=128,index_topk=2048)),
            parallel_config=NS(decode_context_parallel_size=1,prefill_context_parallel_size=1,cp_kv_cache_interleave_size=1),
            compilation_config=NS(static_forward_context={}),
            scheduler_config=NS(max_num_seqs=32,max_num_batched_tokens=8192),
            speculative_config=NS(num_speculative_tokens=5,enable_adaptive_verification=False),num_speculative_tokens=5)

    def builder(self,ratio=4,cls=AmpereIndexerMetadataBuilder):
        spec=MLAAttentionSpec(block_size=128,num_kv_heads=1,head_size=68,dtype=torch.uint8,
                              tokens_per_state=ratio,model_version='deepseek_v41')
        return cls(spec,['layer.index'],self.cfg,torch.device('cpu'),block_table_width=4096)

    def test_original_parent_dispatch_reproduces_reported_failure(self):
        with patch.object(AmpereIndexerMetadataBuilder,'_resolve_indexer_uses_fp4',stock.DeepseekV32IndexerMetadataBuilder._resolve_indexer_uses_fp4):
            with self.assertRaisesRegex(ValueError,'requires Blackwell'):
                self.builder()

    def test_full_ampere_constructor_completes_for_cache_ratios(self):
        for ratio in (1,4,128):
            with self.subTest(ratio=ratio):
                b=self.builder(ratio)
                self.assertTrue(b.indexer_uses_fp4)
                self.assertFalse(b.fused_mqa_topk_eligible)
                self.assertEqual(b.compress_ratio,ratio)
                self.assertEqual(b.decode_threshold,6)
                self.assertEqual(b.expanded_block_table_buffer.shape,(192,4096))
                self.assertTrue(b.use_flattening)
                self.assertFalse(b.supports_varlen)
                self.assertEqual(b.decode_seq_lens_buffer.device.type,'cpu')

    def test_stock_guard_still_rejects_other_unsupported_gpus(self):
        for cap in (80,86,89,90,120):
            with self.subTest(cap=cap):
                self.platform.is_device_capability_family=lambda f,c=cap:f==c
                with self.assertRaisesRegex(ValueError,'requires Blackwell'):
                    stock.dsa_indexer_uses_fp4(self.cfg)
        self.platform.is_device_capability_family=lambda f:f==100
        self.assertTrue(stock.dsa_indexer_uses_fp4(self.cfg))

    def test_stock_builder_still_uses_stock_guard(self):
        with self.assertRaisesRegex(ValueError,'requires Blackwell'):
            self.builder(cls=stock.DeepseekV32IndexerMetadataBuilder)

    def test_ampere_rejects_fp8_cache(self):
        self.cfg.attention_config.indexer_kv_dtype='fp8'
        with self.assertRaisesRegex(ValueError,'MXFP4 indexer cache only'):
            self.builder()

    def test_ampere_builder_rejects_wrong_device(self):
        self.platform.get_device_capability=lambda:DeviceCapability(9,0)
        with self.assertRaisesRegex(ValueError,'requires an SM80 GPU'):
            self.builder()

if __name__=='__main__':unittest.main()
