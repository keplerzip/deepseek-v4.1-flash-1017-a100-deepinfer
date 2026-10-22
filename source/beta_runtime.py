"""Approximate YOCO-consumer FFN compaction for the experimental build.

Producer KV, attention and all tensor coordinates remain full length. Historical
consumer FFN outputs outside the replay tail/checkpoints are zeroed. This is a
deliberate numerical approximation, not full/exact DeepInfer YOCO compaction.
"""
from __future__ import annotations

import os
import logging
import torch


def retained_rows(query_starts, positions, keep=256, stride=8192):
    """Keep each request's chunk tail plus checkpoint tails, including decode."""
    if keep < 128 or stride <= keep:
        raise ValueError("Replay tail must cover SWA and be shorter than checkpoint stride")
    keep_mask = torch.zeros(positions.shape[0], dtype=torch.bool, device=positions.device)
    for start, end in zip(query_starts[:-1], query_starts[1:]):
        start, end = int(start), int(end)
        if not 0 <= start <= end <= positions.numel():
            raise ValueError("Invalid request token boundaries")
        keep_mask[max(start, end - keep):end] = True
    keep_mask |= positions.remainder(stride) >= stride - keep
    return keep_mask.nonzero(as_tuple=False).flatten()


def consumer_ffn(layer, x, input_ids, positions):
    if (os.environ.get("DEEPINFER_BETA_YOCO_FFN", "0") != "1"
            or not 21 <= layer.beta_layer_id < 40 or layer.use_sequence_parallel
            or x.shape[0] <= 256):
        return layer.ffn(x, input_ids)
    from vllm.forward_context import get_forward_context, is_forward_context_available
    if not is_forward_context_available():
        return layer.ffn(x, input_ids)
    context = get_forward_context()
    metadata = context.attn_metadata
    if not isinstance(metadata, dict):
        return layer.ffn(x, input_ids)
    swa = next((m for m in metadata.values()
                if getattr(m, "num_prefills", 0) and hasattr(m, "query_start_loc_cpu")), None)
    if swa is None or swa.query_start_loc_cpu is None:
        return layer.ffn(x, input_ids)
    if x.is_cuda and torch.cuda.is_current_stream_capturing():
        # Only eager prefill compacts. Decode graphs retain fixed coordinates.
        return layer.ffn(x, input_ids)
    count = int(getattr(swa, "num_actual_tokens", positions.numel()))
    starts = swa.query_start_loc_cpu.tolist()
    if starts[-1] != count or count > positions.numel() or x.shape[0] != positions.numel():
        return layer.ffn(x, input_ids)
    indices = retained_rows(starts, positions[:count],
                            int(os.environ.get("DEEPINFER_BETA_YOCO_KEEP", "256")))
    if indices.numel() == x.shape[0]:
        return layer.ffn(x, input_ids)
    padding = getattr(context, "is_padding", None)
    try:
        if padding is not None:
            context.is_padding = padding.index_select(0, indices)
        compact = layer.ffn(x.index_select(0, indices),
                            input_ids.index_select(0, indices) if input_ids is not None else None)
        result = torch.zeros_like(x)
        result.index_copy_(0, indices, compact)
    finally:
        context.is_padding = padding
    layer.beta_compacted_calls = getattr(layer, "beta_compacted_calls", 0) + 1
    layer.beta_skipped_rows = getattr(layer, "beta_skipped_rows", 0) + x.shape[0] - indices.numel()
    if layer.beta_compacted_calls == 1:
        logging.getLogger(__name__).warning(
            "BETA approximate YOCO FFN active: layer=%s kept=%s total=%s",
            layer.beta_layer_id, indices.numel(), x.shape[0])
    return result
