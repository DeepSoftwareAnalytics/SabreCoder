"""
Sparsity utilities.

This repo evaluates *causal* (autoregressive) attention. For sparsity metrics, the meaningful
denominator is the causal-eligible region only (the lower triangle), excluding positions that
are always masked by causality (upper triangle).
"""

from __future__ import annotations

import math


def ceil_div(a: int, b: int) -> int:
    a = int(a)
    b = int(b)
    if b <= 0:
        raise ValueError(f"ceil_div expects b>0, got b={b}")
    return (a + b - 1) // b


def causal_total_positions(seq_len: int) -> int:
    """Total token-level causal-eligible positions for a square (seq_len x seq_len) attention matrix."""
    if seq_len <= 0:
        return 0
    return seq_len * (seq_len + 1) // 2


def causal_total_blocks(num_query_blocks: int, num_key_blocks: int | None = None) -> int:
    """
    Total block-level causal-eligible blocks for an (M x N) block attention matrix.

    Matches the logic used in `methods/sabrecoder/sabrecoder_llama_triton.py`.
    """
    if num_key_blocks is None:
        num_key_blocks = num_query_blocks

    m = int(num_query_blocks)
    n = int(num_key_blocks)
    if m <= 0 or n <= 0:
        return 0

    if m <= n:
        return m * (m + 1) // 2
    # First n rows grow to n, remaining rows are full n.
    return n * (n + 1) // 2 + (m - n) * n


def causal_active_blocks_window_plus_keys(
    seq_len: int,
    window_size: int,
    *,
    key_token_mask=None,
    block_size: int = 64,
) -> tuple[int, int]:
    """
    Exact block-level active block count within the causal-eligible region (lower triangle),
    for patterns of the form:

      attend(query) = sliding_window(query, window_size) OR key_token_mask

    where key_token_mask is a query-independent set of "extra keys" (subject to causality).

    This matches the block-level sparsity accounting used in SabreCoder and the Triton kernels
    that gate blocks based on the *earliest query
    token in the block* (so window blocks per query block are `ceil(window_size/block_size)+1`).

    Args:
        seq_len: token length after truncation (square attention).
        window_size: token window size (0 disables window).
        key_token_mask: 1D bool/0-1 mask of length seq_len, or 2D [1, seq_len], or None.
        block_size: block size (default 64).

    Returns:
        (active_blocks, total_causal_blocks)
    """
    seq_len = int(seq_len)
    if seq_len <= 0:
        return 0, 0

    bs = int(block_size)
    num_blocks = ceil_div(seq_len, bs)
    total_causal_blocks = num_blocks * (num_blocks + 1) // 2

    # Build block-level key mask (query-independent), shape [num_blocks]
    if key_token_mask is None:
        key_block_prefix = [0] * (num_blocks + 1)
    else:
        import torch

        if isinstance(key_token_mask, torch.Tensor):
            km = key_token_mask
            if km.dim() == 2 and km.shape[0] == 1:
                km = km[0]
            if km.dim() != 1:
                raise ValueError(f"key_token_mask must be 1D (or [1, L]), got shape={tuple(km.shape)}")
            km = km.to(torch.bool).cpu()
            if km.numel() != seq_len:
                raise ValueError(f"key_token_mask length {km.numel()} != seq_len {seq_len}")
            idx = torch.nonzero(km, as_tuple=False).flatten()
            if idx.numel() == 0:
                key_block_prefix = [0] * (num_blocks + 1)
            else:
                blk = torch.unique(idx // bs)
                block_mask = torch.zeros(num_blocks, dtype=torch.int32)
                block_mask[blk] = 1
                prefix = torch.cumsum(block_mask, dim=0)
                key_block_prefix = [0] + prefix.tolist()
        else:
            # Assume sequence of bool/int
            if len(key_token_mask) != seq_len:
                raise ValueError(f"key_token_mask length {len(key_token_mask)} != seq_len {seq_len}")
            block_mask = [0] * num_blocks
            for i, v in enumerate(key_token_mask):
                if v:
                    block_mask[i // bs] = 1
            key_block_prefix = [0]
            s = 0
            for v in block_mask:
                s += int(v)
                key_block_prefix.append(s)

    active_blocks = 0
    window_size = int(window_size)
    if window_size <= 0:
        # Only key_token_mask (causal): for query block qb, active key blocks are all key blocks <= qb.
        # Since key_block_prefix is cumulative over blocks, active(qb)=prefix[qb+1].
        for qb in range(num_blocks):
            active_blocks += int(key_block_prefix[qb + 1])
        return int(active_blocks), int(total_causal_blocks)

    span_blocks = ceil_div(window_size, bs)  # how many full blocks the window spans backwards
    for qb in range(num_blocks):
        # Window blocks for this query block: [max(0, qb-span_blocks), qb]
        window_start_block = qb - span_blocks
        if window_start_block < 0:
            window_start_block = 0

        window_len = qb - window_start_block + 1
        extra_key_blocks_before_window = int(key_block_prefix[window_start_block])
        active_blocks += window_len + extra_key_blocks_before_window

    return int(active_blocks), int(total_causal_blocks)


def causal_attended_positions_from_cap(seq_len: int, attend_cap: float) -> float:
    """
    Approximate total attended positions within the causal region by assuming each query attends to
    at most `attend_cap` keys, capped again by causality: attend_i = min(i, attend_cap), i=1..seq_len.

    This is useful for theoretical sparsity estimates derived from a per-query "budget" (window, sink,
    global tokens, etc.), without materializing a mask.
    """
    if seq_len <= 0:
        return 0.0

    cap = max(0.0, min(float(attend_cap), float(seq_len)))
    full = int(math.floor(cap))
    # Sum_{i=1..full} i  +  (seq_len-full) * cap
    return (full * (full + 1) / 2.0) + (seq_len - full) * cap


def causal_attend_ratio_from_cap(seq_len: int, attend_cap: float) -> float:
    """Attend ratio within the causal region, using `causal_attended_positions_from_cap`."""
    denom = float(causal_total_positions(seq_len))
    if denom <= 0.0:
        return 0.0
    return min(max(causal_attended_positions_from_cap(seq_len, attend_cap) / denom, 0.0), 1.0)


def causal_sparsity_from_cap(seq_len: int, attend_cap: float) -> float:
    """Sparsity within the causal region (1 - attend_ratio), using `causal_attend_ratio_from_cap`."""
    return 1.0 - causal_attend_ratio_from_cap(seq_len, attend_cap)
