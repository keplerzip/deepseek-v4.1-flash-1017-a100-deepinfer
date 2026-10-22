"""Project reusable groups onto the pinned native CPU offloading connector.

DSv4.1's compressor ring is per-request scratch, not a prefix checkpoint.
Keep it in the model's original allocator and block tables, but exclude it
from the native offloader. Native hashes, cache policy and DMA are unchanged.
"""
from dataclasses import replace

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector as NativeOffloadingConnector,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.kv_cache_interface import CircularBufferSpec, iter_layer_specs

logger = init_logger(__name__)


class OffloadingConnector(NativeOffloadingConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        self._original_group_count = len(kv_cache_config.kv_cache_groups)
        kept = []
        for index, group in enumerate(kv_cache_config.kv_cache_groups):
            if not group.kv_cache_spec.prefix_cacheable:
                if not all(isinstance(spec, CircularBufferSpec)
                           for spec in iter_layer_specs(group.kv_cache_spec)):
                    raise ValueError('Unsupported non-prefix-cacheable KV group')
            elif group.enable_kv_transfer:
                kept.append(index)
        if not kept:
            raise ValueError('No reusable KV groups for CPU offloading')
        self._group_ids = tuple(kept)
        self._group_index = {original: local for local, original in enumerate(kept)}
        projected = replace(kv_cache_config, kv_cache_groups=[
            kv_cache_config.kv_cache_groups[index] for index in kept])
        if resolve_kv_cache_block_sizes(projected, vllm_config) != resolve_kv_cache_block_sizes(kv_cache_config, vllm_config):
            raise ValueError('CPU offload projection changed scheduler/hash alignment')
        self._layer_names = {name for group in projected.kv_cache_groups
                             for name in group.layer_names}
        super().__init__(vllm_config, role, projected)
        logger.info('R3-HF2 native CPU KV: reusable groups=%s; original groups=%d; '
                    'per-request compressor rings remain local', kept, self._original_group_count)

    def _select(self, groups):
        if groups is None or len(groups) == 0:
            return groups
        if len(groups) != self._original_group_count:
            raise ValueError('Unexpected original KV group count')
        return tuple(groups[index] for index in self._group_ids)

    def _record_allocations(self, groups):
        if groups:
            self.connector_scheduler._current_batch_allocated_block_ids.update(
                block for group in groups for block in group if block != 0)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        # Scratch can reuse a block with a pending native store. Keep the
        # native overwrite fence informed about allocations in ALL groups.
        self._record_allocations(blocks.get_block_ids())
        return super().update_state_after_alloc(
            request, replace(blocks, blocks=self._select(blocks.blocks)), num_external_tokens)

    def build_connector_meta(self, scheduler_output):
        new = []
        for request in scheduler_output.scheduled_new_reqs:
            self._record_allocations(request.block_ids)
            new.append(replace(request, block_ids=self._select(request.block_ids)))
        cached = scheduler_output.scheduled_cached_reqs
        for groups in cached.new_block_ids:
            self._record_allocations(groups)
        cached = replace(cached, new_block_ids=[self._select(groups)
                                             for groups in cached.new_block_ids])
        block_state = scheduler_output.kv_connector_block_state
        if block_state is not None:
            original_resolver = block_state.resolve_block_ids
            handoffs = {req: [(self._group_index[group], block, boundary)
                              for group, block, boundary in entries
                              if group in self._group_index]
                        for req, entries in block_state.boundary_state_offloads.items()}
            block_state = replace(block_state,
                resolve_block_ids=lambda req: self._select(original_resolver(req)),
                boundary_state_offloads=handoffs)
        projected = replace(scheduler_output, scheduled_new_reqs=new,
                            scheduled_cached_reqs=cached, kv_connector_block_state=block_state)
        return super().build_connector_meta(projected)

    def register_kv_caches(self, kv_caches):
        return super().register_kv_caches({name: kv_caches[name]
                                          for name in self._layer_names})
