# SPDX-License-Identifier: Apache-2.0
# Adapted geometry only; retain the existing R2 fused mHC implementation.

def mhc_fused_post_pre_split_config(num_tokens: int, hidden_size: int, hc_mult: int):
    """Retain the A100 backport cutoff/tuning, rejecting incomplete tiles."""
    import os
    from vllm.model_executor.kernels.mhc.tilelang import _SMALL_FMA_CONFIG
    if not 0 < num_tokens <= 16:
        return None
    tile_n, n_splits, n_thr = _SMALL_FMA_CONFIG
    mix_size = hc_mult * (hc_mult + 2)
    if mix_size % tile_n or hidden_size % (n_splits * n_thr):
        # Only the aligned geometry is taken from Deepinfer-SM80. The R2
        # fused auxiliary capture ABI and warmup registration stay intact.
        if (
            os.environ.get("VLLM_DSV41_DEEPINFER_MHC_ALIGNED", "0") == "1"
            and mix_size % 6 == 0
            and hidden_size % (8 * 128) == 0
        ):
            tile_n, n_splits, n_thr = 6, 8, 128
        else:
            tile_n = 2 if num_tokens < 8 else 3
            n_splits = 8 if num_tokens < 8 and hidden_size <= 4096 else 4
            n_thr = 256
    if mix_size % tile_n or hidden_size % (n_splits * n_thr):
        return None
    return tile_n, n_splits, n_thr
