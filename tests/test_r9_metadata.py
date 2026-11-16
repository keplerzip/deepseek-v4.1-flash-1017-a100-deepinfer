"""Native metadata slicing preserves KV-side tensors and exact query bounds."""
import unittest
import torch
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerPrefillChunkMetadata,slice_prefill_chunk_query)


class MetadataReuseTests(unittest.TestCase):
    def test_query_slices_preserve_every_kv_side_field(self):
        chunk=DeepseekV32IndexerPrefillChunkMetadata(
            block_table=torch.arange(8).reshape(2,4),cu_seqlen_ks=torch.arange(20),
            cu_seqlen_ke=torch.arange(20)+5,cu_seq_lens=torch.tensor([0,5,12]),
            token_to_seq=torch.tensor([0]*5+[1]*7),total_seq_lens=12,max_seqlen_k=7,
            token_start=40,token_end=60,num_reqs=2)
        for query in (slice(0,5),slice(5,12),slice(12,20),slice(18,40)):
            with self.subTest(query=query):
                part=slice_prefill_chunk_query(chunk,query)
                self.assertEqual(part.token_start,40+min(query.start,20))
                self.assertEqual(part.token_end,40+min(query.stop,20))
                torch.testing.assert_close(part.cu_seqlen_ks,chunk.cu_seqlen_ks[query])
                torch.testing.assert_close(part.cu_seqlen_ke,chunk.cu_seqlen_ke[query])
                self.assertIs(part.block_table,chunk.block_table)
                self.assertIs(part.cu_seq_lens,chunk.cu_seq_lens)
                self.assertIs(part.token_to_seq,chunk.token_to_seq)
                self.assertEqual(part.total_seq_lens,chunk.total_seq_lens)
                self.assertEqual(part.skip_kv_gather,query.start>0)
        self.assertEqual(chunk.token_start,40)
        self.assertIsNone(slice_prefill_chunk_query(None,slice(0,5)))


if __name__=='__main__':unittest.main()
