import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("beta_compute", ROOT / "source/beta_runtime.py")
beta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(beta)


class BetaComputeTests(unittest.TestCase):
    def test_request_tails_and_checkpoint_tails_preserve_decode_rows(self):
        pos = torch.cat((torch.tensor([14000, 14001, 14002]), torch.arange(7936, 16384)))
        indices = beta.retained_rows([0, 3, len(pos)], pos)
        retained = set(indices.tolist())
        self.assertTrue(set(range(3)).issubset(retained))
        self.assertTrue(set(range(len(pos)-256, len(pos))).issubset(retained))
        self.assertTrue(set(range(3, 259)).issubset(retained))
        self.assertLess(len(retained), len(pos))

    def test_invalid_replay_window_is_not_accepted(self):
        with self.assertRaises(ValueError):
            beta.retained_rows([0, 512], torch.arange(512), keep=64)

    def test_default_and_explicit_off_run_full_consumer_ffn(self):
        for flag in (None, '0'):
            with self.subTest(flag=flag), patch.dict(os.environ):
                if flag is None:
                    os.environ.pop('DEEPINFER_BETA_YOCO_FFN', None)
                else:
                    os.environ['DEEPINFER_BETA_YOCO_FFN'] = flag
                sizes = []
                def ffn(x, ids):
                    sizes.append(len(x))
                    return x + 2
                layer = SimpleNamespace(beta_layer_id=25, use_sequence_parallel=False, ffn=ffn)
                x = torch.ones(4096, 8)
                result = beta.consumer_ffn(layer, x, torch.arange(4096), torch.arange(4096))
                self.assertEqual(sizes, [4096])
                self.assertTrue(torch.equal(result, x + 2))
                self.assertFalse(hasattr(layer, 'beta_skipped_rows'))

    def test_compacted_ffn_runs_only_retained_rows_and_restores_padding(self):
        positions = torch.arange(4096)
        metadata = SimpleNamespace(num_prefills=1, num_actual_tokens=4096,
                                   query_start_loc_cpu=torch.tensor([0,4096]))
        padding = torch.zeros(4096, dtype=torch.bool)
        context = SimpleNamespace(attn_metadata={"swa":metadata}, is_padding=padding)
        sizes = []
        def ffn(x, ids):
            sizes.append(len(x))
            self.assertEqual(len(context.is_padding), len(x))
            return x + 2
        layer = SimpleNamespace(beta_layer_id=25, use_sequence_parallel=False, ffn=ffn)
        with patch('vllm.forward_context.is_forward_context_available', return_value=True), patch(
                'vllm.forward_context.get_forward_context', return_value=context), patch.dict(
                    os.environ, {'DEEPINFER_BETA_YOCO_FFN':'1'}):
            result = beta.consumer_ffn(layer, torch.ones(4096,8), positions, positions)
        self.assertEqual(sizes, [256])
        self.assertTrue(torch.equal(result[-256:], torch.full((256,8),3.)))
        self.assertEqual(result[:-256].count_nonzero(), 0)
        self.assertIs(context.is_padding, padding)
        self.assertEqual(layer.beta_skipped_rows, 3840)

    def test_producer_and_decode_do_not_compact(self):
        for layer_id, rows in [(20,4096), (25,6), (40,4096)]:
            x = torch.ones(rows,8)
            layer = SimpleNamespace(beta_layer_id=layer_id, use_sequence_parallel=False, ffn=lambda x, ids:x+1)
            self.assertTrue(torch.equal(beta.consumer_ffn(layer,x,None,torch.arange(rows)),x+1))

    def test_bitshift_scales_all_256_codes_match_f32_and_bf16(self):
        codes = torch.arange(256,dtype=torch.int32)
        bits = codes << 23
        bits[codes == 0] = 1 << 22
        values = bits.view(torch.float32)
        reference = torch.pow(torch.tensor(2.,dtype=torch.float64), codes.double()-127).float()
        self.assertTrue(torch.equal(values, reference))
        bf_bits = (codes << 7).to(torch.int16)
        bf_bits[codes == 0] = 64
        self.assertTrue(torch.equal(bf_bits.view(torch.bfloat16),reference.to(torch.bfloat16)))

    def test_source_preserves_existing_8192_scan_inverse_rope_and_blocked_prefill(self):
        source = (ROOT / 'source/beta-overlay/files/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py').read_text()
        self.assertIn('_MAX_ONE_BLOCK_INDPTR_ROWS = 8192', source)
        self.assertIn('FUSE_INV_ROPE=fuse_inv_rope', source)
        self.assertIn('def _sparse_attn_prefill_blocked_kernel(', source)
        self.assertIn('def _beta_reference_decode_partial_kernel(', source)
        self.assertIn('import os', source)
