"""SabreCoder Triton wrapper for LLaMA attention."""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import time

from sabrecoder_attention_triton import StructureAwareSparseAttentionTriton


# Global state
_SPARSE_CALL_COUNT = 0
_DENSE_CALL_COUNT = 0
_CURRENT_CODE = None
_SPARSITY_STATS = {
    # NOTE: We track sparsity w.r.t. the *causal-eligible* region only (block-triangular),
    # i.e., denominator excludes blocks that are always masked by causal attention.
    'total_blocks': 0,  # Number of causal-eligible blocks observed so far.
    'active_blocks': 0,  # Number of attended blocks within the causal-eligible region.
    'num_samples': 0,
    'enabled': False
}

# Global attention timing statistics
_ATTENTION_TIMING = {
    'total_time': 0.0,        # Total attention computation time (seconds)
    'call_count': 0,          # Number of attention calls
    'sparse_time': 0.0,       # Time spent in sparse attention
    'dense_time': 0.0,        # Time spent in dense attention
    'sparse_calls': 0,        # Number of sparse attention calls
    'dense_calls': 0,         # Number of dense attention calls
    'enabled': False,         # Whether timing is enabled
}

# Optional cache for precomputed active block metadata from the evaluator.
_PRECOMPUTED_BLOCK_INDICES = None
_PRECOMPUTED_BLOCK_COUNTS = None


def set_current_code(code: str):
    global _CURRENT_CODE
    _CURRENT_CODE = code


def get_current_code() -> Optional[str]:
    global _CURRENT_CODE
    return _CURRENT_CODE


def clear_current_code():
    global _CURRENT_CODE
    _CURRENT_CODE = None


def set_precomputed_blocks(block_indices, block_counts):
    """Store precomputed active-block metadata for reuse during attention calls."""
    global _PRECOMPUTED_BLOCK_INDICES, _PRECOMPUTED_BLOCK_COUNTS
    _PRECOMPUTED_BLOCK_INDICES = block_indices
    _PRECOMPUTED_BLOCK_COUNTS = block_counts


def get_precomputed_blocks():
    """Return cached precomputed active-block metadata."""
    global _PRECOMPUTED_BLOCK_INDICES, _PRECOMPUTED_BLOCK_COUNTS
    return _PRECOMPUTED_BLOCK_INDICES, _PRECOMPUTED_BLOCK_COUNTS


def clear_precomputed_blocks():
    """Clear cached precomputed active-block metadata."""
    global _PRECOMPUTED_BLOCK_INDICES, _PRECOMPUTED_BLOCK_COUNTS
    _PRECOMPUTED_BLOCK_INDICES = None
    _PRECOMPUTED_BLOCK_COUNTS = None


def get_attention_call_counts():
    return {'sparse': _SPARSE_CALL_COUNT, 'dense': _DENSE_CALL_COUNT}


def reset_attention_call_counts():
    global _SPARSE_CALL_COUNT, _DENSE_CALL_COUNT
    _SPARSE_CALL_COUNT = 0
    _DENSE_CALL_COUNT = 0


def enable_attention_timing(enabled: bool = True):
    """Enable or disable attention timing."""
    global _ATTENTION_TIMING
    _ATTENTION_TIMING['enabled'] = enabled
    if enabled:
        _ATTENTION_TIMING['total_time'] = 0.0
        _ATTENTION_TIMING['call_count'] = 0
        _ATTENTION_TIMING['sparse_time'] = 0.0
        _ATTENTION_TIMING['dense_time'] = 0.0
        _ATTENTION_TIMING['sparse_calls'] = 0
        _ATTENTION_TIMING['dense_calls'] = 0


def get_attention_timing():
    """Return the current attention timing statistics."""
    stats = _ATTENTION_TIMING.copy()
    if stats['call_count'] > 0:
        stats['avg_time'] = stats['total_time'] / stats['call_count']
    else:
        stats['avg_time'] = 0.0

    if stats['sparse_calls'] > 0:
        stats['avg_sparse_time'] = stats['sparse_time'] / stats['sparse_calls']
    else:
        stats['avg_sparse_time'] = 0.0

    if stats['dense_calls'] > 0:
        stats['avg_dense_time'] = stats['dense_time'] / stats['dense_calls']
    else:
        stats['avg_dense_time'] = 0.0

    return stats


def reset_attention_timing():
    """Reset attention timing statistics."""
    global _ATTENTION_TIMING
    _ATTENTION_TIMING['total_time'] = 0.0
    _ATTENTION_TIMING['call_count'] = 0
    _ATTENTION_TIMING['sparse_time'] = 0.0
    _ATTENTION_TIMING['dense_time'] = 0.0
    _ATTENTION_TIMING['sparse_calls'] = 0
    _ATTENTION_TIMING['dense_calls'] = 0


def enable_sparsity_tracking(enabled: bool = True):
    """Enable or disable block-level sparsity tracking."""
    global _SPARSITY_STATS
    _SPARSITY_STATS['enabled'] = enabled
    if enabled:
        _SPARSITY_STATS['total_blocks'] = 0
        _SPARSITY_STATS['active_blocks'] = 0
        _SPARSITY_STATS['num_samples'] = 0


def update_sparsity_stats(block_mask: torch.Tensor):
    """Update global sparsity statistics from a block mask."""
    global _SPARSITY_STATS
    if not _SPARSITY_STATS['enabled']:
        return

    if block_mask.dim() != 2:
        raise ValueError(f"update_sparsity_stats expects a 2D block_mask, got shape={tuple(block_mask.shape)}")

    num_blocks_m, num_blocks_n = block_mask.shape

    if num_blocks_m <= num_blocks_n:
        total_blocks = num_blocks_m * (num_blocks_m + 1) // 2
    else:
        total_blocks = num_blocks_n * (num_blocks_n + 1) // 2 + (num_blocks_m - num_blocks_n) * num_blocks_n

    active_blocks = block_mask.tril(diagonal=0).sum().item()

    _SPARSITY_STATS['total_blocks'] += total_blocks
    _SPARSITY_STATS['active_blocks'] += active_blocks
    _SPARSITY_STATS['num_samples'] += 1


def update_sparsity_stats_from_blocks(total_blocks: int, active_blocks: int):
    """Update global sparsity statistics from precomputed block counts."""
    global _SPARSITY_STATS
    if not _SPARSITY_STATS['enabled']:
        return
    
    _SPARSITY_STATS['total_blocks'] += int(total_blocks)
    _SPARSITY_STATS['active_blocks'] += int(active_blocks)
    _SPARSITY_STATS['num_samples'] += 1


def get_sparsity_stats():
    """Return the current block-level sparsity statistics."""
    stats = _SPARSITY_STATS.copy()
    if stats['total_blocks'] > 0:
        stats['attend_ratio'] = stats['active_blocks'] / stats['total_blocks']
        stats['sparsity_ratio'] = 1.0 - stats['attend_ratio']
    else:
        stats['sparsity_ratio'] = 0.0
        stats['attend_ratio'] = 0.0
    return stats


class StructureAwareAttentionWrapperTriton(nn.Module):
    """LLaMA attention wrapper backed by SabreCoder Triton kernels."""

    def __init__(
        self,
        original_attn,
        tokenizer,
        use_sparse_for_generation: bool = False,
        block_size: int = 64,
        window_size: int = 64,
        num_prefix_tokens: int = 128,
        num_suffix_tokens: int = 256,
        use_token_level_sparsity: bool = False,
        embedding_weight: Optional[torch.Tensor] = None,
        global_last_k_chunks: int = 2,
        max_chunk_tokens: int = 0,
        use_chunk_similarity: bool = False,
        chunk_similarity_top_percent: float = 0.1,
        chunk_similarity_max_tokens_per_chunk: int = 256,
        chunk_similarity_max_neighbors: int = 8,
        use_crossfile_chunk_similarity: bool = False,
        crossfile_chunk_similarity_top_percent: Optional[float] = None,
    ):
        super().__init__()
        self.original_attn = original_attn
        self.tokenizer = tokenizer
        self.use_sparse_for_generation = use_sparse_for_generation
        self.window_size = window_size
        self.num_prefix_tokens = num_prefix_tokens
        self.num_suffix_tokens = num_suffix_tokens
        self.use_token_level_sparsity = use_token_level_sparsity

        # Copy attributes
        self.config = original_attn.config
        self.layer_idx = getattr(original_attn, 'layer_idx', 0)
        self.head_dim = original_attn.head_dim
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = original_attn.num_key_value_groups
        self.scaling = original_attn.scaling
        self.attention_dropout = original_attn.attention_dropout
        self.is_causal = True

        # Triton sparse attention
        self.sparse_attn = StructureAwareSparseAttentionTriton(
            tokenizer=tokenizer,
            block_size=block_size,
            window_size=window_size,
            num_prefix_tokens=num_prefix_tokens,
            num_suffix_tokens=num_suffix_tokens,
            use_token_level_sparsity=use_token_level_sparsity,
            embedding_weight=embedding_weight,
            global_last_k_chunks=global_last_k_chunks,
            max_chunk_tokens=max_chunk_tokens,
            use_chunk_similarity=use_chunk_similarity,
            chunk_similarity_top_percent=chunk_similarity_top_percent,
            chunk_similarity_max_tokens_per_chunk=chunk_similarity_max_tokens_per_chunk,
            chunk_similarity_max_neighbors=chunk_similarity_max_neighbors,
            use_crossfile_chunk_similarity=use_crossfile_chunk_similarity,
            crossfile_chunk_similarity_top_percent=crossfile_chunk_similarity_top_percent,
        )

        # Projections
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj

    def _dense_attention_fast(self, query_states, key_states, value_states, attention_mask=None):
        """Fast dense attention for generation - minimal overhead"""
        # In decode mode, query length is usually 1 while key length includes KV cache.
        query_len = query_states.shape[2]
        key_len = key_states.shape[2]

        # Enable built-in causal masking only for square attention without an explicit mask.
        use_causal = (attention_mask is None and query_len == key_len)

        return torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=use_causal,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        batch_size, seq_len = hidden_states.shape[:2]

        is_generation_phase = (seq_len == 1)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        if is_generation_phase and not self.use_sparse_for_generation:
            global _DENSE_CALL_COUNT
            _DENSE_CALL_COUNT += 1
            attn_output = self._dense_attention_fast(query_states, key_states, value_states, attention_mask)
        else:
            global _SPARSE_CALL_COUNT
            _SPARSE_CALL_COUNT += 1
            code = get_current_code()
            attn_output = self.sparse_attn(query_states, key_states, value_states, code=code, attention_mask=attention_mask)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


def patch_model_with_sabrecoder_attention_triton(
    model,
    tokenizer,
    use_sparse_for_generation: bool = False,
    block_size: int = 64,
    window_size: int = 64,
    num_prefix_tokens: int = 128,
    num_suffix_tokens: int = 256,
    use_token_level_sparsity: bool = False,
    global_last_k_chunks: int = 2,
    max_chunk_tokens: int = 0,
    use_chunk_similarity: bool = False,
    chunk_similarity_top_percent: float = 0.1,
    chunk_similarity_max_tokens_per_chunk: int = 256,
    chunk_similarity_max_neighbors: int = 8,
    use_crossfile_chunk_similarity: bool = False,
    crossfile_chunk_similarity_top_percent: Optional[float] = None,
):
    """Replace LLaMA attention modules with SabreCoder Triton attention."""
    mode = "Full Sparse" if use_sparse_for_generation else "Hybrid (Sparse Prefill, Dense Generation)"
    sparsity_mode = "Token-level" if use_token_level_sparsity else "Block-level"
    print(f"Applying SabreCoder Triton attention [{mode}]")
    print(f"  Sparsity mode: {sparsity_mode}")
    print(f"  Block size: {block_size}")
    print(f"  Window size: {window_size}")
    print(f"  Prefix tokens: {num_prefix_tokens}")
    print(f"  Suffix tokens: {num_suffix_tokens}")
    print(f"  Global last-k chunks: {global_last_k_chunks}")
    print(f"  Max chunk tokens: {max_chunk_tokens}")

    embedding_layer = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    embedding_weight = getattr(embedding_layer, "weight", None) if embedding_layer is not None else None
    if use_chunk_similarity and embedding_weight is None:
        raise ValueError("use_chunk_similarity=True requires model.get_input_embeddings().weight to be available.")

    count = 0
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'self_attn'):
            layer.self_attn = StructureAwareAttentionWrapperTriton(
                layer.self_attn,
                tokenizer=tokenizer,
                use_sparse_for_generation=use_sparse_for_generation,
                block_size=block_size,
                window_size=window_size,
                num_prefix_tokens=num_prefix_tokens,
                num_suffix_tokens=num_suffix_tokens,
                use_token_level_sparsity=use_token_level_sparsity,
                embedding_weight=embedding_weight,
                global_last_k_chunks=global_last_k_chunks,
                max_chunk_tokens=max_chunk_tokens,
                use_chunk_similarity=use_chunk_similarity,
                chunk_similarity_top_percent=chunk_similarity_top_percent,
                chunk_similarity_max_tokens_per_chunk=chunk_similarity_max_tokens_per_chunk,
                chunk_similarity_max_neighbors=chunk_similarity_max_neighbors,
                use_crossfile_chunk_similarity=use_crossfile_chunk_similarity,
                crossfile_chunk_similarity_top_percent=crossfile_chunk_similarity_top_percent,
            )
            count += 1

    print(f"Replaced {count} attention layers with Triton sparse attention")

    model._sparse_attn_enabled = True
    model._sparse_attn_config = {
        'type': 'sabrecoder_triton',
        'use_sparse_for_generation': use_sparse_for_generation,
        'block_size': block_size,
        'window_size': window_size,
        'num_prefix_tokens': num_prefix_tokens,
        'num_suffix_tokens': num_suffix_tokens,
        'use_token_level_sparsity': use_token_level_sparsity,
        'global_last_k_chunks': global_last_k_chunks,
        'max_chunk_tokens': max_chunk_tokens,
        'use_chunk_similarity': use_chunk_similarity,
        'chunk_similarity_top_percent': chunk_similarity_top_percent,
        'chunk_similarity_max_tokens_per_chunk': chunk_similarity_max_tokens_per_chunk,
        'chunk_similarity_max_neighbors': chunk_similarity_max_neighbors,
        'use_crossfile_chunk_similarity': use_crossfile_chunk_similarity,
        'crossfile_chunk_similarity_top_percent': crossfile_chunk_similarity_top_percent,
    }

    return model


if __name__ == "__main__":
    print("SabreCoder Triton attention module")
