"""
SabreCoder block-sparse attention with Triton kernels.

Pipeline:
1. Build chunk connectivity from code structure, references, similarity, and global tokens.
2. Map chunk relations to active block indices.
3. Run Triton kernels over the active causal blocks only.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional, Tuple, List, Dict, Set
import sys
import os
from dataclasses import replace
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semantic_chunker import TreeSitterSegmenter, ChunkType, CodeChunk
from reference_graph import CrossReferenceAnalyzer


# ==================== Triton Kernel ====================

@triton.jit
def _fwd_kernel_sabrecoder(
    Q, K, V, Out,
    BlockIndices,  # Active block indices: (batch, num_blocks_M, max_active_blocks)
    BlockCounts,   # Number of active blocks per query block: (batch, num_blocks_M)
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_bi_b, stride_bi_m, stride_bi_k,
    stride_bc_b, stride_bc_m,
    batch, n_heads, seq_len,
    max_active_blocks: tl.constexpr,  # Compile-time upper bound of active key blocks per query block.
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Block-sparse forward kernel over precomputed active block indices."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // n_heads
    off_h = off_hz % n_heads

    q_offset = off_b * stride_qb + off_h * stride_qh
    k_offset = off_b * stride_kb + off_h * stride_kh
    v_offset = off_b * stride_vb + off_h * stride_vh
    o_offset = off_b * stride_ob + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len, other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    # Scaling factor
    scale = 1.0 / tl.sqrt(float(BLOCK_DMODEL))

    count_ptr = BlockCounts + off_b * stride_bc_b + start_m * stride_bc_m
    num_active = tl.load(count_ptr)

    for i in range(max_active_blocks):
        should_process = i < num_active

        if should_process:
            idx_ptr = BlockIndices + off_b * stride_bi_b + start_m * stride_bi_m + i * stride_bi_k
            block_n = tl.load(idx_ptr)

            start_n = block_n * BLOCK_N
            offs_n = start_n + tl.arange(0, BLOCK_N)

            valid_mask = (offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len)
            causal_mask = offs_n[None, :] <= offs_m[:, None]
            valid_mask = valid_mask & causal_mask

            k_ptrs = K + k_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
            v_ptrs = V + v_offset + offs_n[None, :] * stride_vn + offs_d[:, None] * stride_vk

            k = tl.load(k_ptrs, mask=offs_n[None, :] < seq_len, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[None, :] < seq_len, other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            qk *= scale

            qk = tl.where(valid_mask, qk, float("-inf"))

            # Online softmax
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            m_ij = tl.where(m_ij == float("-inf"), float(0.0), m_ij)

            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)

            alpha = tl.exp(m_i - m_ij)
            acc = acc * alpha[:, None]
            acc += tl.dot(p.to(v.dtype), tl.trans(v))

            l_i = l_i * alpha + l_ij
            m_i = m_ij

    l_i = tl.where(l_i == 0.0, float(1.0), l_i)
    acc = acc / l_i[:, None]

    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seq_len)


@triton.jit
def _fwd_kernel_sabrecoder_token_sparse(
    Q, K, V, Out,
    BlockIndices,  # Active block indices: (batch, num_blocks_M, max_active_blocks)
    BlockCounts,   # Number of active blocks per query block: (batch, num_blocks_M)
    TokenMask,     # Token-level mask: (seq_len, seq_len)
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_bi_b, stride_bi_m, stride_bi_k,
    stride_bc_b, stride_bc_m,
    stride_tm, stride_tn,
    batch, n_heads, seq_len,
    max_active_blocks: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Internal utility function."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // n_heads
    off_h = off_hz % n_heads

    q_offset = off_b * stride_qb + off_h * stride_qh
    k_offset = off_b * stride_kb + off_h * stride_kh
    v_offset = off_b * stride_vb + off_h * stride_vh
    o_offset = off_b * stride_ob + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len, other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    # Scaling factor
    scale = 1.0 / tl.sqrt(float(BLOCK_DMODEL))

    count_ptr = BlockCounts + off_b * stride_bc_b + start_m * stride_bc_m
    num_active = tl.load(count_ptr)

    for i in range(max_active_blocks):
        should_process = i < num_active

        if should_process:
            idx_ptr = BlockIndices + off_b * stride_bi_b + start_m * stride_bi_m + i * stride_bi_k
            block_n = tl.load(idx_ptr)

            start_n = block_n * BLOCK_N
            offs_n = start_n + tl.arange(0, BLOCK_N)

            valid_mask = (offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len)
            causal_mask = offs_n[None, :] <= offs_m[:, None]
            valid_mask = valid_mask & causal_mask

            mask_ptrs = TokenMask + offs_m[:, None] * stride_tm + offs_n[None, :] * stride_tn
            token_mask = tl.load(mask_ptrs, mask=valid_mask, other=float("-inf"))
            
            structure_mask = (token_mask == 0.0)
            valid_mask = valid_mask & structure_mask

            k_ptrs = K + k_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
            v_ptrs = V + v_offset + offs_n[None, :] * stride_vn + offs_d[:, None] * stride_vk

            k = tl.load(k_ptrs, mask=offs_n[None, :] < seq_len, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[None, :] < seq_len, other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            qk *= scale

            qk = tl.where(valid_mask, qk, float("-inf"))

            # Online softmax
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            m_ij = tl.where(m_ij == float("-inf"), float(0.0), m_ij)

            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)

            alpha = tl.exp(m_i - m_ij)
            acc = acc * alpha[:, None]
            acc += tl.dot(p.to(v.dtype), tl.trans(v))

            l_i = l_i * alpha + l_ij
            m_i = m_ij

    l_i = tl.where(l_i == 0.0, float(1.0), l_i)
    acc = acc / l_i[:, None]

    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seq_len)


# ==================== Mask Builder ====================

_GLOBAL_CHUNK_CACHE = {}
_GLOBAL_LINE_TO_TOKEN_CACHE = {}
_GLOBAL_MASK_CACHE = {}
_GLOBAL_SIM_EDGES_CACHE = {}
_GLOBAL_SIM_VEC_CACHE = {}
_GLOBAL_CACHE_MAX_SIZE = 50
_GLOBAL_SIM_VEC_CACHE_MAX_SIZE = 8


class StructureAwareMaskBuilder:
    """Internal utility function."""

    def __init__(
        self,
        tokenizer,
        window_size=64,
        num_prefix_tokens=128,
        num_suffix_tokens=256,
        block_size: int = 64,
        global_last_k_chunks: int = 2,
        max_chunk_tokens: int = 0,
        embedding_weight: Optional[torch.Tensor] = None,
        use_chunk_similarity: bool = False,
        chunk_similarity_top_percent: float = 0.1,
        chunk_similarity_max_tokens_per_chunk: int = 256,
        chunk_similarity_max_neighbors: int = 8,
        use_crossfile_chunk_similarity: bool = False,
        crossfile_chunk_similarity_top_percent: Optional[float] = None,
    ):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.num_prefix_tokens = num_prefix_tokens
        self.num_suffix_tokens = num_suffix_tokens
        self.block_size = int(block_size)
        self.global_last_k_chunks = int(global_last_k_chunks)
        self.max_chunk_tokens = int(max_chunk_tokens)
        self.embedding_weight = embedding_weight
        self.use_chunk_similarity = bool(use_chunk_similarity)
        self.chunk_similarity_top_percent = float(chunk_similarity_top_percent)
        self.chunk_similarity_max_tokens_per_chunk = int(chunk_similarity_max_tokens_per_chunk)
        self.chunk_similarity_max_neighbors = int(chunk_similarity_max_neighbors)
        self.use_crossfile_chunk_similarity = bool(use_crossfile_chunk_similarity)
        self.crossfile_chunk_similarity_top_percent = (
            float(crossfile_chunk_similarity_top_percent)
            if crossfile_chunk_similarity_top_percent is not None
            else float(chunk_similarity_top_percent)
        )

        if self.block_size <= 0:
            raise ValueError("block_size must be > 0.")
        if self.global_last_k_chunks < 0:
            raise ValueError("global_last_k_chunks must be >= 0.")
        if self.max_chunk_tokens < 0:
            raise ValueError("max_chunk_tokens must be >= 0.")
        if self.max_chunk_tokens > 0 and (self.max_chunk_tokens % self.block_size) != 0:
            raise ValueError("max_chunk_tokens must be a multiple of block_size to align with kernel blocks.")

        if self.use_crossfile_chunk_similarity and not self.use_chunk_similarity:
            raise ValueError("use_crossfile_chunk_similarity=True requires use_chunk_similarity=True.")

        if self.use_chunk_similarity:
            if self.embedding_weight is None:
                raise ValueError("use_chunk_similarity=True requires embedding_weight (model.get_input_embeddings().weight).")
            # Accept either fraction [0, 1] or "percent" (e.g. 10 means 10%). Allow 0 to mean "no edges".
            if self.chunk_similarity_top_percent < 0.0:
                raise ValueError("chunk_similarity_top_percent must be >= 0.")
            if self.chunk_similarity_max_tokens_per_chunk <= 0:
                raise ValueError("chunk_similarity_max_tokens_per_chunk must be > 0.")
            # Allow 0 to mean "no edges".
            if self.chunk_similarity_max_neighbors < 0:
                raise ValueError("chunk_similarity_max_neighbors must be >= 0.")
            if self.use_crossfile_chunk_similarity and self.crossfile_chunk_similarity_top_percent < 0.0:
                raise ValueError("crossfile_chunk_similarity_top_percent must be >= 0.")

    def build_token_level_mask_multi_files(
        self,
        files: List[Dict],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        header_template: str = "// FILE: {path}\\n",
    ) -> torch.Tensor:
        """Internal utility function."""
        global _GLOBAL_MASK_CACHE, _GLOBAL_CACHE_MAX_SIZE

        segmenter = TreeSitterSegmenter(tokenizer=self.tokenizer)

        combined_chunks: List = []
        combined_chunk_deps: Dict[int, Set[int]] = {}
        combined_import_chunks: Set[int] = set()
        combined_parts: List[str] = []

        line_offset = 0
        chunk_offset = 0

        for f in files:
            path = f.get('path', '<unknown>')
            code = f.get('code', '')
            language = f.get('language', None)

            header = header_template.format(path=path)
            combined_parts.append(header)
            header_lines = header.count('\\n')
            line_offset += header_lines

            chunks = segmenter.segment_code(code, language=language)
            import_chunks = {i for i, c in enumerate(chunks) if c.chunk_type == ChunkType.IMPORT}

            local_deps: Dict[int, Set[int]] = {}
            analyzer = CrossReferenceAnalyzer(chunks, code, language=language)
            refs = analyzer.analyze()
            for ref in refs:
                src, tgt = ref.source_chunk_idx, ref.target_chunk_idx
                for idx in (src, tgt):
                    local_deps.setdefault(idx, set())
                local_deps[src].add(tgt)
                local_deps[tgt].add(src)

            for idx, ch in enumerate(chunks):
                new_chunk = replace(
                    ch,
                    start_line=ch.start_line + line_offset,
                    end_line=ch.end_line + line_offset,
                    signature_start_line=(ch.signature_start_line + line_offset) if ch.signature_start_line else None,
                    signature_end_line=(ch.signature_end_line + line_offset) if ch.signature_end_line else None,
                )
                combined_chunks.append(new_chunk)

            for src, tgts in local_deps.items():
                combined_src = src + chunk_offset
                combined_chunk_deps.setdefault(combined_src, set())
                for tgt in tgts:
                    combined_chunk_deps[combined_src].add(tgt + chunk_offset)

            for imp_idx in import_chunks:
                combined_import_chunks.add(imp_idx + chunk_offset)

            chunk_offset += len(chunks)
            num_lines = code.count('\\n') + (0 if code.endswith('\\n') else 1 if code else 0)
            line_offset += num_lines

            combined_parts.append(code)
            if not code.endswith('\\n'):
                combined_parts.append('\\n')

        combined_code = ''.join(combined_parts)

        code_hash = hash(combined_code)
        cache_key = (
            "multi_file",
            code_hash,
            seq_len,
            device,
            dtype,
            self.block_size,
            self.global_last_k_chunks,
            self.max_chunk_tokens,
            self.use_chunk_similarity,
            self.chunk_similarity_top_percent,
            self.chunk_similarity_max_tokens_per_chunk,
            self.chunk_similarity_max_neighbors,
            self.use_crossfile_chunk_similarity,
            self.crossfile_chunk_similarity_top_percent,
        )
        if cache_key in _GLOBAL_MASK_CACHE:
            return _GLOBAL_MASK_CACHE[cache_key].clone()

        mask = self._build_mask_from_chunks(
            combined_code,
            seq_len,
            device,
            dtype,
            combined_chunks,
            combined_chunk_deps,
            combined_import_chunks,
            language=None,
        )

        causal = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)
        mask[causal] = float('-inf')

        if len(_GLOBAL_MASK_CACHE) >= _GLOBAL_CACHE_MAX_SIZE:
            _GLOBAL_MASK_CACHE.pop(next(iter(_GLOBAL_MASK_CACHE)))
        _GLOBAL_MASK_CACHE[cache_key] = mask.clone()

        return mask

    def _get_token_to_line_mapping(self, code: str, seq_len: int):
        """Internal utility function."""
        global _GLOBAL_LINE_TO_TOKEN_CACHE

        code_hash = hash(code)
        cache_key = (code_hash, seq_len)

        if cache_key in _GLOBAL_LINE_TO_TOKEN_CACHE:
            return _GLOBAL_LINE_TO_TOKEN_CACHE[cache_key]

        lines = code.split('\n')
        char_to_line = []
        for line_idx, line in enumerate(lines):
            char_to_line.extend([line_idx + 1] * (len(line) + 1))

        encoded = self.tokenizer(
            code,
            add_special_tokens=True,
            truncation=True,
            max_length=seq_len,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping", None)
        input_ids = encoded.get("input_ids", None)
        if offsets is None or input_ids is None:
            raise RuntimeError("Tokenizer must support return_offsets_mapping and return input_ids for token mapping.")

        token_to_line = [0] * seq_len
        n = min(seq_len, len(input_ids))
        for token_idx in range(n):
            start_char, _ = offsets[token_idx]
            if start_char is None or start_char < 0:
                continue
            if not char_to_line:
                token_to_line[token_idx] = 1
                continue
            char_pos = min(start_char, len(char_to_line) - 1)
            token_to_line[token_idx] = char_to_line[char_pos]

        _GLOBAL_LINE_TO_TOKEN_CACHE[cache_key] = token_to_line
        return token_to_line

    def _map_lines_to_tokens(self, code: str, line_numbers: set, seq_len: int) -> torch.Tensor:
        """Internal utility function."""
        if not line_numbers:
            return torch.zeros(seq_len, dtype=torch.bool)

        token_to_line = self._get_token_to_line_mapping(code, seq_len)
        token_mask = torch.tensor([line_num in line_numbers for line_num in token_to_line], dtype=torch.bool)
        return token_mask

    def _clean_for_similarity(self, text: str) -> str:
        """
        For similarity embedding only:
        - Crossfile segments are typically stored as commented code lines ("# " / "// ").
        - Strip these prefixes to avoid comments dominating embedding similarity.
        """
        out = []
        for ln in text.split("\n"):
            if ln.startswith("# "):
                out.append(ln[2:])
            elif ln.startswith("// "):
                out.append(ln[3:])
            else:
                out.append(ln)
        return "\n".join(out)

    def _chunk_vector(self, text: str) -> Optional[torch.Tensor]:
        if not text or not text.strip():
            return None
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            return None
        if len(ids) > self.chunk_similarity_max_tokens_per_chunk:
            ids = ids[: self.chunk_similarity_max_tokens_per_chunk]

        emb = self.embedding_weight
        if emb is None:
            raise RuntimeError("embedding_weight is required for chunk similarity but is None.")
        ids_t = torch.tensor(ids, device=emb.device, dtype=torch.long)
        with torch.no_grad():
            return emb.index_select(0, ids_t).to(dtype=torch.float32).mean(dim=0)

    def _get_or_build_similarity_vectors(self, chunks: List, code: str) -> tuple[torch.Tensor, List[int]]:
        """
        Return (mat, idx_map) where:
          - mat: (M, D) normalized vectors for non-empty chunks (float16 for cache, on emb.device)
          - idx_map: length M list mapping row -> original chunk idx
        """
        if self.embedding_weight is None:
            raise RuntimeError("embedding_weight is required for chunk similarity but is None.")

        global _GLOBAL_SIM_VEC_CACHE, _GLOBAL_SIM_VEC_CACHE_MAX_SIZE
        code_hash = hash(code)
        emb = self.embedding_weight
        tok_key = getattr(self.tokenizer, "name_or_path", None) or self.tokenizer.__class__.__name__
        vec_key = (
            code_hash,
            tok_key,
            int(self.chunk_similarity_max_tokens_per_chunk),
            str(emb.device),
            str(emb.dtype),
            int(emb.data_ptr()),
        )
        cached = _GLOBAL_SIM_VEC_CACHE.get(vec_key)
        if cached is not None:
            mat, idx_map = cached
            return mat, list(idx_map)

        texts: List[str] = []
        idx_map: List[int] = []
        for i, c in enumerate(chunks):
            txt = getattr(c, "text", "") or ""
            cleaned = self._clean_for_similarity(txt)
            if cleaned and cleaned.strip():
                texts.append(cleaned)
                idx_map.append(i)

        if not texts:
            # Cache the empty result to avoid repeated work on pathological inputs.
            if len(_GLOBAL_SIM_VEC_CACHE) >= _GLOBAL_SIM_VEC_CACHE_MAX_SIZE:
                _GLOBAL_SIM_VEC_CACHE.pop(next(iter(_GLOBAL_SIM_VEC_CACHE)))
            _GLOBAL_SIM_VEC_CACHE[vec_key] = (torch.empty((0, 0), device=emb.device, dtype=torch.float16), [])
            return _GLOBAL_SIM_VEC_CACHE[vec_key][0], []

        # Prefer batched tokenization + one embedding gather for speed.
        # Fall back to per-chunk encode if the tokenizer does not support the required interface.
        try:
            encoded = self.tokenizer(
                texts,
                add_special_tokens=False,
                truncation=True,
                max_length=int(self.chunk_similarity_max_tokens_per_chunk),
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded.get("input_ids", None)
            attn_mask = encoded.get("attention_mask", None)
            if input_ids is None or attn_mask is None:
                raise RuntimeError("Tokenizer did not return input_ids/attention_mask for batched call.")
        except Exception:
            vecs: List[torch.Tensor] = []
            for txt in texts:
                v = self._chunk_vector(txt)
                if v is None:
                    # Keep alignment: represent empty as zeros so indices remain stable.
                    v = torch.zeros((emb.shape[1],), device=emb.device, dtype=torch.float32)
                vecs.append(v)
            mat32 = torch.stack(vecs, dim=0).to(dtype=torch.float32)
            mat32 = torch.nn.functional.normalize(mat32, dim=1)
            mat = mat32.to(dtype=torch.float16)
        else:
            device = emb.device
            input_ids = input_ids.to(device=device, dtype=torch.long)
            attn_mask = attn_mask.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                flat = input_ids.reshape(-1)
                flat_emb = emb.index_select(0, flat).to(dtype=torch.float32)
                bsz, seqlen = input_ids.shape
                dim = int(flat_emb.shape[-1])
                token_emb = flat_emb.view(bsz, seqlen, dim)
                mask = attn_mask.unsqueeze(-1)  # (B, L, 1)
                summed = (token_emb * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                mat32 = summed / denom
                mat32 = torch.nn.functional.normalize(mat32, dim=1)
                mat = mat32.to(dtype=torch.float16)

        # Cache (LRU)
        if len(_GLOBAL_SIM_VEC_CACHE) >= _GLOBAL_SIM_VEC_CACHE_MAX_SIZE:
            _GLOBAL_SIM_VEC_CACHE.pop(next(iter(_GLOBAL_SIM_VEC_CACHE)))
        _GLOBAL_SIM_VEC_CACHE[vec_key] = (mat, list(idx_map))
        return mat, list(idx_map)

    def _similarity_edges(
        self,
        chunks: List,
        code: str,
        *,
        src_indices: Optional[List[int]] = None,
        tgt_indices: Optional[List[int]] = None,
        top_percent: Optional[float] = None,
    ) -> Dict[int, Set[int]]:
        """
        Compute chunk similarity edges using token embedding lookup only (no forward).

        NOTE: Using "global top-% of all pairs" quickly becomes O(N^2) edges as the number of chunks grows,
        which destroys sparsity. To keep sparsity meaningful and deterministic, we build **directed**
        edges using **per-source top-k neighbors**:
          k = floor(top_percent * num_targets), capped by chunk_similarity_max_neighbors

        IMPORTANT: We intentionally allow k=0. Otherwise, a very small top_percent (e.g. 0.001) would still
        force at least one neighbor per source chunk, which can easily activate many 64x64 blocks and make
        block-level sparsity collapse.
        """
        if not self.use_chunk_similarity:
            return {}

        # Early exit before any tokenization/embedding work.
        top_frac = float(self.chunk_similarity_top_percent) if top_percent is None else float(top_percent)
        # Accept either fraction [0, 1] or "percent" (e.g. 10 means 10%).
        if top_frac > 1.0:
            top_frac = top_frac / 100.0
        top_frac = max(0.0, min(1.0, top_frac))
        if self.chunk_similarity_max_neighbors <= 0 or top_frac <= 0.0:
            return {}

        global _GLOBAL_SIM_EDGES_CACHE, _GLOBAL_CACHE_MAX_SIZE
        code_hash = hash(code)
        if src_indices is None:
            src_indices = list(range(len(chunks)))
        if tgt_indices is None:
            tgt_indices = list(range(len(chunks)))

        sim_key = (
            code_hash,
            float(top_frac),
            self.chunk_similarity_max_tokens_per_chunk,
            self.chunk_similarity_max_neighbors,
            hash(tuple(src_indices)),
            hash(tuple(tgt_indices)),
        )
        if sim_key in _GLOBAL_SIM_EDGES_CACHE:
            cached = _GLOBAL_SIM_EDGES_CACHE[sim_key]
            return {k: set(v) for k, v in cached.items()}

        mat16, idx_map = self._get_or_build_similarity_vectors(chunks, code)
        if not idx_map:
            return {}

        # Map original chunk index -> vector row index
        chunk_to_vec = {chunk_idx: vec_idx for vec_idx, chunk_idx in enumerate(idx_map)}
        src_vec_rows = [(src, chunk_to_vec[src]) for src in src_indices if src in chunk_to_vec]
        tgt_vec_rows = [(tgt, chunk_to_vec[tgt]) for tgt in tgt_indices if tgt in chunk_to_vec]
        if not src_vec_rows or not tgt_vec_rows:
            return {}

        src_chunks, src_rows = zip(*src_vec_rows)
        tgt_chunks, tgt_rows = zip(*tgt_vec_rows)

        mat = mat16.to(dtype=torch.float32)

        src_mat = mat[list(src_rows)]  # (S, D)
        tgt_mat = mat[list(tgt_rows)]  # (T, D)
        sims = src_mat @ tgt_mat.t()   # (S, T)

        # Exclude self-similarity when src is also in targets
        tgt_pos = {c: j for j, c in enumerate(tgt_chunks)}
        for i, c in enumerate(src_chunks):
            j = tgt_pos.get(c, None)
            if j is not None:
                sims[i, j] = float("-inf")

        edges: Dict[int, Set[int]] = {}
        num_tgt = len(tgt_chunks)
        if num_tgt == 0:
            return {}

        k_from_percent = int(math.floor(num_tgt * top_frac))
        k = min(int(self.chunk_similarity_max_neighbors), k_from_percent)
        if k <= 0:
            return {}

        top_idx = torch.topk(sims, k=min(k, num_tgt), dim=1, largest=True, sorted=False).indices
        for i, src_chunk in enumerate(src_chunks):
            for j in top_idx[i].tolist():
                tgt_chunk = int(tgt_chunks[j])
                if tgt_chunk == int(src_chunk):
                    continue
                edges.setdefault(int(src_chunk), set()).add(tgt_chunk)

        # Cache (LRU)
        if len(_GLOBAL_SIM_EDGES_CACHE) >= _GLOBAL_CACHE_MAX_SIZE:
            _GLOBAL_SIM_EDGES_CACHE.pop(next(iter(_GLOBAL_SIM_EDGES_CACHE)))
        _GLOBAL_SIM_EDGES_CACHE[sim_key] = {k: set(v) for k, v in edges.items()}
        return edges

    def _is_python_code(self, code: str) -> bool:
        """Internal utility function."""
        non_python_markers = [
            'using System', 'using (',
            'namespace ', 'public class', 'private class',
            'public static void', '#include <', 'package ',
            'function ', 'var ', 'let ', 'const ',
        ]
        for marker in non_python_markers:
            if marker in code:
                return False
        return True

    def _identify_special_lines(self, code: str, chunks, language: Optional[str] = None) -> dict:
        """Internal utility function."""
        import re

        if language is None:
            # Avoid language guessing for debuggability; callers should pass an explicit language.
            return {'global_vars': set(), 'type_aliases': set(), 'class_attrs': set(), 'return_stmts': set()}

        lang = str(language).lower()
        if lang not in {"python", "py"}:
            return {'global_vars': set(), 'type_aliases': set(), 'class_attrs': set(), 'return_stmts': set()}

        lines = code.split('\n')
        result = {
            'global_vars': set(),
            'type_aliases': set(),
            'class_attrs': set(),
            'return_stmts': set()
        }

        function_class_lines = set()
        init_method_lines = set()

        for chunk in chunks:
            if chunk.chunk_type in [ChunkType.FUNCTION, ChunkType.CLASS, ChunkType.METHOD]:
                function_class_lines.update(range(chunk.start_line, chunk.end_line + 1))

            if chunk.chunk_type == ChunkType.METHOD and chunk.name == '__init__':
                init_method_lines.update(range(chunk.start_line, chunk.end_line + 1))

        for line_idx, line in enumerate(lines):
            line_num = line_idx + 1
            stripped = line.strip()

            if not stripped or stripped.startswith('#'):
                continue

            if line_num not in function_class_lines:
                if '=' in stripped and not any(kw in stripped for kw in ['==', '!=', '<=', '>=', 'def ', 'class ', 'import ', 'from ']):
                    if re.match(r'^[A-Za-z_][A-Za-z0-9_]*\s*[:\[]?\s*=', stripped):
                        result['global_vars'].add(line_num)

            if 'TypeAlias' in stripped:
                result['type_aliases'].add(line_num)
            elif re.match(r'^[A-Z][A-Za-z0-9_]*\s*=\s*[A-Z][A-Za-z0-9_]*\[', stripped):
                result['type_aliases'].add(line_num)

            if line_num in init_method_lines:
                if re.match(r'^\s*self\.[A-Za-z_][A-Za-z0-9_]*\s*=', line):
                    result['class_attrs'].add(line_num)

            if line_num in function_class_lines:
                if re.match(r'^\s*return\b', stripped):
                    result['return_stmts'].add(line_num)

        return result

    def _append_uncovered_line_chunks(self, code: str, chunks: List) -> List:
        """
        Ensure every line belongs to some chunk.

        This prevents token->chunk mapping from collapsing (many -1) when tree-sitter returns only
        structural nodes (e.g. methods/classes) and leaves top-level / loose lines uncovered.

        New chunks are appended (indices of existing chunks stay stable).
        """
        if not code:
            return chunks

        lines = code.split('\n')
        num_lines = len(lines)
        if num_lines <= 0:
            return chunks

        covered = [False] * (num_lines + 1)  # 1-based
        for ch in chunks or []:
            s = max(1, int(getattr(ch, "start_line", 1) or 1))
            e = min(num_lines, int(getattr(ch, "end_line", 0) or 0))
            if e < s:
                continue
            for ln in range(s, e + 1):
                covered[ln] = True

        out = list(chunks or [])
        ln = 1
        while ln <= num_lines:
            if covered[ln]:
                ln += 1
                continue
            start_ln = ln
            while ln <= num_lines and not covered[ln]:
                ln += 1
            end_ln = ln - 1
            seg_text = "\n".join(lines[start_ln - 1:end_ln])
            out.append(
                CodeChunk(
                    chunk_type=ChunkType.MODULE_CODE,
                    start=0,
                    end=len(seg_text),
                    text=seg_text,
                    name="",
                    parent_name=None,
                    start_line=start_ln,
                    end_line=end_ln,
                    signature_start_line=None,
                    signature_end_line=None,
                )
            )
        return out

    def _map_tokens_to_chunks(self, code: str, chunks, seq_len: int):
        """Internal utility function."""
        token_to_chunk = [-1] * seq_len

        if not chunks:
            return token_to_chunk

        lines = code.split('\n')
        num_lines = len(lines)
        line_to_chunk = [-1] * (num_lines + 1)

        for chunk_idx, chunk in enumerate(chunks):
            for line_num in range(chunk.start_line, chunk.end_line + 1):
                if line_num <= num_lines:
                    line_to_chunk[line_num] = chunk_idx

        # Require offset_mapping for deterministic, debuggable behavior.
        encoded = self.tokenizer(
            code,
            add_special_tokens=True,
            truncation=True,
            max_length=seq_len,
            return_offsets_mapping=True
        )
        offsets = encoded.get("offset_mapping", None)
        input_ids = encoded.get("input_ids", None)
        if offsets is None or input_ids is None:
            raise RuntimeError("Tokenizer must support return_offsets_mapping and return input_ids for token mapping.")
        if len(offsets) < len(input_ids):
            raise RuntimeError("Tokenizer returned offset_mapping shorter than input_ids; cannot build stable mapping.")

        if not input_ids:
            return token_to_chunk

        char_to_line = []
        for line_idx, line in enumerate(lines):
            char_to_line.extend([line_idx + 1] * (len(line) + 1))
        if char_to_line:
            char_to_line[-1] = char_to_line[-1]  # keep last element for bound safety

        for token_idx in range(min(seq_len, len(input_ids))):
            if offsets is not None:
                start_char, _ = offsets[token_idx]
                if start_char is None or start_char < 0:
                    continue
                char_pos = min(start_char, len(char_to_line) - 1) if char_to_line else 0

            line_num = char_to_line[char_pos] if char_to_line else 1
            if line_num < len(line_to_chunk):
                token_to_chunk[token_idx] = line_to_chunk[line_num]

        return token_to_chunk

    def _build_mask_with_crossfile(
        self,
        full_code: str,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        crossfile_code: str,
        current_code: str,
        language: Optional[str] = None,
        crossfile_full_attention: bool = True,
    ) -> torch.Tensor:
        """Run sparse attention when block metadata is available, otherwise fall back to dense attention."""
        if not full_code.startswith(crossfile_code):
            raise ValueError("full_code does not start with crossfile_code; dataset fields are inconsistent.")

        # Cache (include the mode switch to avoid mixing behaviors).
        code_hash = hash(full_code)
        cache_key = (
            "crossfile",
            code_hash,
            seq_len,
            device,
            dtype,
            bool(crossfile_full_attention),
            (str(language).lower() if language else None),
            self.block_size,
            self.global_last_k_chunks,
            self.max_chunk_tokens,
            self.use_chunk_similarity,
            self.chunk_similarity_top_percent,
            self.chunk_similarity_max_tokens_per_chunk,
            self.chunk_similarity_max_neighbors,
            self.use_crossfile_chunk_similarity,
            self.crossfile_chunk_similarity_top_percent,
        )
        if cache_key in _GLOBAL_MASK_CACHE:
            return _GLOBAL_MASK_CACHE[cache_key].clone()

        # Build chunks:
        # - crossfile: either one big chunk (dense mode) or per-segment chunks (chunked mode)
        # - current: tree-sitter chunks, line-shifted
        crossfile_num_lines = len(crossfile_code.split("\n")) if crossfile_code else 0

        chunks: List = []
        import_chunks: Set[int] = set()
        # Extra globally-visible chunks (used to implement dense crossfile keys deterministically via chunk ids).
        global_chunks: Set[int] = set()
        chunk_deps: Dict[int, Set[int]] = {}

        # Crossfile chunks (line-level, do not rely on tree-sitter across multiple files).
        if crossfile_code and crossfile_code.strip():
            if crossfile_full_attention:
                # One large chunk to make crossfile self-attention dense (within-chunk visibility).
                chunks.append(
                    CodeChunk(
                        chunk_type=ChunkType.MODULE_CODE,
                        start=0,
                        end=len(crossfile_code),
                        text=crossfile_code,
                        name="",
                        parent_name=None,
                        start_line=1,
                        end_line=crossfile_num_lines,
                        signature_start_line=None,
                        signature_end_line=None,
                    )
                )
                # Also make crossfile keys globally visible to all queries (including current-file queries).
                global_chunks.add(len(chunks) - 1)
            else:
                # Each retrieved segment in crossfile_context becomes a chunk.
                # Segment headers are produced by data/cceval_rag/build_prompts.py:
                #   "# path: start - end" (python) or "// path: start - end" (java)
                lines = crossfile_code.split("\n")
                header_prefixes = ("# ", "// ")

                def _is_header(s: str) -> bool:
                    s = s.rstrip()
                    if not s.startswith(header_prefixes):
                        return False
                    # Must contain " : <int> - <int>" pattern somewhere near the end.
                    # Keep it strict enough to avoid treating commented code lines as headers.
                    import re
                    return re.match(r"^(#|//)\s+.+:\s*\d+\s*-\s*\d+\s*$", s) is not None

                headers = [i for i, ln in enumerate(lines, start=1) if _is_header(ln)]
                boundaries: List[tuple[int, int]] = []

                if headers:
                    # Pre-header region (rare): include if non-blank, and attach blank lines inside it.
                    if headers[0] > 1:
                        boundaries.append((1, headers[0] - 1))
                    for a, b in zip(headers, headers[1:] + [len(lines) + 1]):
                        boundaries.append((a, b - 1))
                else:
                    boundaries.append((1, len(lines)))

                def _all_blank(s: int, e: int) -> bool:
                    return all(not lines[i - 1].strip() for i in range(s, e + 1))

                # Merge blank-only spans into adjacent non-blank spans so we never create a blank-only chunk
                # and never leave any lines uncovered.
                merged: List[tuple[int, int]] = []
                pending_blank: Optional[tuple[int, int]] = None
                for s, e in boundaries:
                    if e < s:
                        continue
                    if _all_blank(s, e):
                        if merged:
                            prev_s, prev_e = merged[-1]
                            merged[-1] = (prev_s, max(prev_e, e))
                        else:
                            pending_blank = (s, e) if pending_blank is None else (pending_blank[0], max(pending_blank[1], e))
                        continue
                    if pending_blank is not None:
                        s = min(s, pending_blank[0])
                        pending_blank = None
                    merged.append((s, e))

                for start_line, end_line in merged:
                    seg_text = "\n".join(lines[start_line - 1:end_line])
                    chunks.append(
                        CodeChunk(
                            chunk_type=ChunkType.MODULE_CODE,
                            start=0,
                            end=len(seg_text),
                            text=seg_text,
                            name="",
                            parent_name=None,
                            start_line=start_line,
                            end_line=end_line,
                            signature_start_line=None,
                            signature_end_line=None,
                        )
                    )

        crossfile_chunk_count = len(chunks)

        # Current chunks via tree-sitter, shift line numbers
        if current_code and current_code.strip():
            segmenter = TreeSitterSegmenter(tokenizer=self.tokenizer)
            current_chunks = segmenter.segment_code(current_code, language=language)
            current_chunks_filled = self._append_uncovered_line_chunks(current_code, current_chunks)
            shifted_chunks = []
            for c in current_chunks_filled:
                shifted_chunks.append(
                    replace(
                        c,
                        start_line=c.start_line + crossfile_num_lines,
                        end_line=c.end_line + crossfile_num_lines,
                        signature_start_line=(c.signature_start_line + crossfile_num_lines) if c.signature_start_line else None,
                        signature_end_line=(c.signature_end_line + crossfile_num_lines) if c.signature_end_line else None,
                    )
                )
            chunks.extend(shifted_chunks)

            # Dependencies only within current file (shift indices).
            analyzer = CrossReferenceAnalyzer(current_chunks, current_code, language=language)
            references = analyzer.analyze()
            for ref in references:
                src = ref.source_chunk_idx + crossfile_chunk_count
                tgt = ref.target_chunk_idx + crossfile_chunk_count
                chunk_deps.setdefault(src, set()).add(tgt)
                chunk_deps.setdefault(tgt, set()).add(src)

            current_imports = {i for i, c in enumerate(current_chunks) if c.chunk_type == ChunkType.IMPORT}
            import_chunks = {i + crossfile_chunk_count for i in current_imports}

        # Similarity edges: for crossfile prompts, only connect CURRENT -> CROSSFILE by default
        # to avoid O(N^2) crossfile-crossfile densification.
        if self.use_chunk_similarity:
            src_idx = list(range(crossfile_chunk_count, len(chunks)))
            tgt_idx = list(range(0, crossfile_chunk_count))
            sim_deps = self._similarity_edges(chunks, full_code, src_indices=src_idx, tgt_indices=tgt_idx)
            for src, tgts in sim_deps.items():
                chunk_deps.setdefault(src, set()).update(tgts)

            if self.use_crossfile_chunk_similarity and crossfile_chunk_count > 1:
                cf_idx = list(range(0, crossfile_chunk_count))
                sim_cf = self._similarity_edges(
                    chunks,
                    full_code,
                    src_indices=cf_idx,
                    tgt_indices=cf_idx,
                    top_percent=self.crossfile_chunk_similarity_top_percent,
                )
                for src, tgts in sim_cf.items():
                    chunk_deps.setdefault(src, set()).update(tgts)

        # Build base sparse mask from chunks
        merged_import_chunks = set(import_chunks)
        if global_chunks:
            merged_import_chunks.update(global_chunks)
        mask = self._build_mask_from_chunks(
            full_code, seq_len, device, dtype, chunks, chunk_deps, merged_import_chunks, language=language
        )

        # Fail loudly if token-to-chunk mapping is broken (debuggability > silent fallback).
        token_to_chunk = self._map_tokens_to_chunks(full_code, chunks, seq_len)
        valid_mappings = sum(1 for x in token_to_chunk if x != -1)
        mapping_ratio = valid_mappings / seq_len if seq_len > 0 else 0.0
        if mapping_ratio < 0.1:
            raise RuntimeError(
                f"[crossfile_mask] Token-to-chunk mapping ratio too low ({mapping_ratio:.3f}). "
                f"seq_len={seq_len}, valid_mappings={valid_mappings}, num_chunks={len(chunks)}"
            )

        # Note: Dense crossfile keys are implemented via global_chunks (chunk ids) before token mapping,
        # rather than via a separate offset-derived crossfile token mask. This avoids accidental densification
        # when running in chunked mode.

        # Apply causal constraint
        causal = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)
        mask[causal] = float('-inf')

        # Cache (LRU)
        if len(_GLOBAL_MASK_CACHE) >= _GLOBAL_CACHE_MAX_SIZE:
            _GLOBAL_MASK_CACHE.pop(next(iter(_GLOBAL_MASK_CACHE)))
        _GLOBAL_MASK_CACHE[cache_key] = mask.clone()

        return mask

    def _build_sparse_mask_for_code(
        self,
        code: str,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        language: Optional[str] = None,
    ) -> torch.Tensor:
        """Internal utility function."""
        segmenter = TreeSitterSegmenter(tokenizer=self.tokenizer)
        chunks = segmenter.segment_code(code, language=language)
        import_chunks = {i for i, c in enumerate(chunks) if c.chunk_type == ChunkType.IMPORT}

        chunk_deps: Dict[int, Set[int]] = {}
        analyzer = CrossReferenceAnalyzer(chunks, code, language=language)
        references = analyzer.analyze()
        for ref in references:
            src, tgt = ref.source_chunk_idx, ref.target_chunk_idx
            for idx in [src, tgt]:
                if idx not in chunk_deps:
                    chunk_deps[idx] = set()
            chunk_deps[src].add(tgt)
            chunk_deps[tgt].add(src)

        # Ensure full line coverage for stable token->chunk mapping.
        chunks = self._append_uncovered_line_chunks(code, chunks)

        return self._build_mask_from_chunks(
            code, seq_len, device, dtype, chunks, chunk_deps, import_chunks, language=language
        )

    def _apply_chunk_deps_fast(
        self,
        mask: torch.Tensor,
        token_to_chunk_tensor: torch.Tensor,
        chunk_deps: Dict[int, Set[int]],
        *,
        num_chunks: int,
    ) -> None:
        """
        Apply chunk dependency visibility without constructing seq_len x seq_len boolean outer-products per edge.

        The old implementation built `dep_mask = src_mask[:,None] & tgt_mask[None,:]` which materializes a full
        (seq_len, seq_len) boolean tensor for every edge, becoming prohibitively slow once similarity adds many edges.
        """
        if not chunk_deps or num_chunks <= 0:
            return

        device = token_to_chunk_tensor.device
        valid = token_to_chunk_tensor >= 0
        tok_idx = torch.nonzero(valid, as_tuple=True)[0]
        if tok_idx.numel() == 0:
            return

        chunk_idx = token_to_chunk_tensor[tok_idx]
        order = torch.argsort(chunk_idx)
        chunk_sorted = chunk_idx[order]
        tok_sorted = tok_idx[order]

        uniq, counts = torch.unique_consecutive(chunk_sorted, return_counts=True)
        chunk_tokens = [torch.empty((0,), device=device, dtype=torch.long) for _ in range(num_chunks)]
        offset = 0
        for u, c in zip(uniq.tolist(), counts.tolist()):
            u_i = int(u)
            c_i = int(c)
            if 0 <= u_i < num_chunks:
                chunk_tokens[u_i] = tok_sorted[offset:offset + c_i]
            offset += c_i

        for src_idx, tgt_indices in chunk_deps.items():
            if not (0 <= int(src_idx) < num_chunks):
                continue
            src_tokens = chunk_tokens[int(src_idx)]
            if src_tokens.numel() == 0:
                continue

            tgt_list = []
            for tgt_idx in tgt_indices:
                if not (0 <= int(tgt_idx) < num_chunks):
                    continue
                t = chunk_tokens[int(tgt_idx)]
                if t.numel() > 0:
                    tgt_list.append(t)
            if not tgt_list:
                continue

            tgt_tokens = torch.cat(tgt_list, dim=0)
            mask[src_tokens[:, None], tgt_tokens[None, :]] = 0

    def _build_mask_from_chunks(
        self,
        code: str,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        chunks: List,
        chunk_deps: Dict[int, Set[int]],
        import_chunks: Set[int],
        language: Optional[str] = None,
    ) -> torch.Tensor:
        """Internal utility function."""
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype)
        token_to_chunk = self._map_tokens_to_chunks(code, chunks, seq_len)
        token_to_chunk_tensor = torch.tensor(token_to_chunk, device=device, dtype=torch.long)

        window_size = self.window_size
        positions = torch.arange(seq_len, device=device)
        distance = positions.unsqueeze(1) - positions.unsqueeze(0)
        sliding_window_mask = (distance >= 0) & (distance <= window_size)
        mask[sliding_window_mask] = 0

        # Optional: further split "same-chunk" visibility by block-aligned token groups to avoid huge chunks
        # destroying block sparsity (e.g. a giant function body).
        chunk_i = token_to_chunk_tensor.unsqueeze(1)
        chunk_j = token_to_chunk_tensor.unsqueeze(0)
        if self.max_chunk_tokens > 0:
            blocks_per_group = max(1, self.max_chunk_tokens // self.block_size)
            group_idx = (positions // self.block_size) // blocks_per_group  # global, block-aligned
            num_groups = int(group_idx.max().item()) + 2
            group_id = token_to_chunk_tensor * num_groups + group_idx
            group_i = group_id.unsqueeze(1)
            group_j = group_id.unsqueeze(0)
            same_group = (group_i == group_j) & (chunk_i != -1) & (chunk_j != -1)
            mask[same_group] = 0
        else:
            same_chunk = (chunk_i == chunk_j) & (chunk_i != -1)
            mask[same_chunk] = 0

        if chunk_deps:
            self._apply_chunk_deps_fast(mask, token_to_chunk_tensor, chunk_deps, num_chunks=len(chunks))

        globally_visible_lines = set()

        if import_chunks:
            for import_idx in import_chunks:
                import_mask = (token_to_chunk_tensor == import_idx)
                if import_mask.any():
                    mask[:, import_mask] = 0

        for chunk in chunks:
            if chunk.chunk_type in [ChunkType.FUNCTION, ChunkType.CLASS, ChunkType.METHOD]:
                if hasattr(chunk, 'signature_start_line') and hasattr(chunk, 'signature_end_line'):
                    if chunk.signature_start_line and chunk.signature_end_line:
                        globally_visible_lines.update(
                            range(chunk.signature_start_line, chunk.signature_end_line + 1)
                        )

        special_lines = self._identify_special_lines(code, chunks, language=language)
        globally_visible_lines.update(special_lines['global_vars'])
        globally_visible_lines.update(special_lines['type_aliases'])
        globally_visible_lines.update(special_lines['class_attrs'])
        globally_visible_lines.update(special_lines['return_stmts'])

        if globally_visible_lines:
            global_token_mask = self._map_lines_to_tokens(code, globally_visible_lines, seq_len)
            global_token_mask = global_token_mask.to(device)
            if global_token_mask.any():
                mask[:, global_token_mask] = 0

        num_prefix_global = min(self.num_prefix_tokens, seq_len)
        if num_prefix_global > 0:
            mask[:, :num_prefix_global] = 0

        num_suffix_global = min(self.num_suffix_tokens, seq_len)
        if num_suffix_global > 0:
            suffix_start = max(0, seq_len - num_suffix_global)
            mask[:, suffix_start:] = 0

        k = int(self.global_last_k_chunks)
        if k > 0 and len(chunks) > 0:
            start = max(0, len(chunks) - k)
            for chunk_idx in range(start, len(chunks)):
                chunk_mask = (token_to_chunk_tensor == chunk_idx)
                if chunk_mask.any():
                    mask[:, chunk_mask] = 0

        # If mapping ratio is too low, fail loudly (do not silently switch to dense).
        valid_mappings = (token_to_chunk_tensor != -1).sum().item()
        mapping_ratio = valid_mappings / seq_len if seq_len > 0 else 0
        if mapping_ratio < 0.1:
            raise RuntimeError(
                f"Token-to-chunk mapping ratio too low ({mapping_ratio:.3f}). "
                f"seq_len={seq_len}, valid_mappings={valid_mappings}, num_chunks={len(chunks)}"
            )

        return mask

    def build_token_level_mask(
        self,
        code: str,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        crossfile_code: Optional[str] = None,
        current_code: Optional[str] = None,
        language: Optional[str] = None,
        crossfile_full_attention: bool = True,
    ) -> torch.Tensor:
        """Internal utility function."""
        global _GLOBAL_MASK_CACHE, _GLOBAL_CHUNK_CACHE, _GLOBAL_CACHE_MAX_SIZE

        if crossfile_code is not None and current_code is not None:
            return self._build_mask_with_crossfile(
                code,
                seq_len,
                device,
                dtype,
                crossfile_code,
                current_code,
                language=language,
                crossfile_full_attention=crossfile_full_attention,
            )

        code_hash = hash(code)
        mask_cache_key = (
            code_hash,
            seq_len,
            device,
            dtype,
            (str(language).lower() if language else None),
            self.block_size,
            self.global_last_k_chunks,
            self.max_chunk_tokens,
            self.use_chunk_similarity,
            self.chunk_similarity_top_percent,
            self.chunk_similarity_max_tokens_per_chunk,
            self.chunk_similarity_max_neighbors,
            self.use_crossfile_chunk_similarity,
            self.crossfile_chunk_similarity_top_percent,
        )

        if mask_cache_key in _GLOBAL_MASK_CACHE:
            return _GLOBAL_MASK_CACHE[mask_cache_key].clone()


        lang_key = (str(language).lower() if language else None)
        chunk_cache_key = (code_hash, lang_key)
        if chunk_cache_key not in _GLOBAL_CHUNK_CACHE:
            chunks = []
            chunk_deps = {}
            import_chunks = set()

            segmenter = TreeSitterSegmenter(tokenizer=self.tokenizer)
            chunks = segmenter.segment_code(code, language=language)
            import_chunks = {i for i, c in enumerate(chunks) if c.chunk_type == ChunkType.IMPORT}

            analyzer = CrossReferenceAnalyzer(chunks, code, language=language)
            references = analyzer.analyze()

            for ref in references:
                src, tgt = ref.source_chunk_idx, ref.target_chunk_idx
                for idx in [src, tgt]:
                    if idx not in chunk_deps:
                        chunk_deps[idx] = set()
                chunk_deps[src].add(tgt)
                chunk_deps[tgt].add(src)

            # Ensure full line coverage for stable token->chunk mapping.
            chunks = self._append_uncovered_line_chunks(code, chunks)

            _GLOBAL_CHUNK_CACHE[chunk_cache_key] = {
                'chunks': chunks,
                'chunk_deps': chunk_deps,
                'import_chunks': import_chunks
            }

        cached = _GLOBAL_CHUNK_CACHE[chunk_cache_key]
        chunks = cached['chunks']
        chunk_deps = cached['chunk_deps']
        import_chunks = cached['import_chunks']

        # Merge similarity edges (do not mutate cached deps)
        if self.use_chunk_similarity:
            sim_deps = self._similarity_edges(chunks, code)
            if sim_deps:
                merged = {k: set(v) for k, v in chunk_deps.items()} if chunk_deps else {}
                for src, tgts in sim_deps.items():
                    merged.setdefault(src, set()).update(tgts)
                chunk_deps = merged

        mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype)
        token_to_chunk = self._map_tokens_to_chunks(code, chunks, seq_len)
        token_to_chunk_tensor = torch.tensor(token_to_chunk, device=device, dtype=torch.long)

        window_size = self.window_size
        positions = torch.arange(seq_len, device=device)
        distance = positions.unsqueeze(1) - positions.unsqueeze(0)
        sliding_window_mask = (distance >= 0) & (distance <= window_size)
        mask[sliding_window_mask] = 0

        chunk_i = token_to_chunk_tensor.unsqueeze(1)
        chunk_j = token_to_chunk_tensor.unsqueeze(0)
        if self.max_chunk_tokens > 0:
            blocks_per_group = max(1, self.max_chunk_tokens // self.block_size)
            group_idx = (positions // self.block_size) // blocks_per_group
            num_groups = int(group_idx.max().item()) + 2
            group_id = token_to_chunk_tensor * num_groups + group_idx
            group_i = group_id.unsqueeze(1)
            group_j = group_id.unsqueeze(0)
            same_group = (group_i == group_j) & (chunk_i != -1) & (chunk_j != -1)
            mask[same_group] = 0
        else:
            same_chunk = (chunk_i == chunk_j) & (chunk_i != -1)
            mask[same_chunk] = 0

        if chunk_deps:
            self._apply_chunk_deps_fast(mask, token_to_chunk_tensor, chunk_deps, num_chunks=len(chunks))

        globally_visible_lines = set()

        # Import chunks
        if import_chunks:
            for import_idx in import_chunks:
                import_mask = (token_to_chunk_tensor == import_idx)
                if import_mask.any():
                    mask[:, import_mask] = 0

        for chunk in chunks:
            if chunk.chunk_type in [ChunkType.FUNCTION, ChunkType.CLASS, ChunkType.METHOD]:
                if hasattr(chunk, 'signature_start_line') and hasattr(chunk, 'signature_end_line'):
                    if chunk.signature_start_line and chunk.signature_end_line:
                        globally_visible_lines.update(
                            range(chunk.signature_start_line, chunk.signature_end_line + 1)
                        )

        special_lines = self._identify_special_lines(code, chunks, language=language)
        globally_visible_lines.update(special_lines['global_vars'])
        globally_visible_lines.update(special_lines['type_aliases'])
        globally_visible_lines.update(special_lines['class_attrs'])
        globally_visible_lines.update(special_lines['return_stmts'])

        if globally_visible_lines:
            global_token_mask = self._map_lines_to_tokens(code, globally_visible_lines, seq_len)
            global_token_mask = global_token_mask.to(device)
            if global_token_mask.any():
                mask[:, global_token_mask] = 0

        num_prefix_global = min(self.num_prefix_tokens, seq_len)
        if num_prefix_global > 0:
            mask[:, :num_prefix_global] = 0

        num_suffix_global = min(self.num_suffix_tokens, seq_len)
        if num_suffix_global > 0:
            suffix_start = max(0, seq_len - num_suffix_global)
            mask[:, suffix_start:] = 0

        k = int(self.global_last_k_chunks)
        if k > 0 and len(chunks) > 0:
            start = max(0, len(chunks) - k)
            for chunk_idx in range(start, len(chunks)):
                chunk_mask = (token_to_chunk_tensor == chunk_idx)
                if chunk_mask.any():
                    mask[:, chunk_mask] = 0

        # If mapping ratio is too low, fail loudly (do not silently switch to dense).
        valid_mappings = (token_to_chunk_tensor != -1).sum().item()
        mapping_ratio = valid_mappings / seq_len if seq_len > 0 else 0

        if mapping_ratio < 0.1:
            raise RuntimeError(
                f"Token-to-chunk mapping ratio too low ({mapping_ratio:.3f}). "
                f"seq_len={seq_len}, valid_mappings={valid_mappings}, num_chunks={len(chunks)}"
            )

        causal = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)
        mask[causal] = float('-inf')

        if len(_GLOBAL_MASK_CACHE) >= _GLOBAL_CACHE_MAX_SIZE:
            _GLOBAL_MASK_CACHE.pop(next(iter(_GLOBAL_MASK_CACHE)))
        _GLOBAL_MASK_CACHE[mask_cache_key] = mask.clone()

        return mask

    def _build_block_mask_from_chunks(
        self,
        *,
        code: str,
        seq_len: int,
        chunks: List,
        chunk_deps: Dict[int, Set[int]],
        import_chunks: Set[int],
        language: Optional[str],
    ) -> torch.Tensor:
        """
        Build a **block-level** mask directly (avoid materializing seq_len x seq_len token masks).

        Semantics match `compute_block_mask_from_token_mask(build_token_level_mask(...))`:
        a block (i, j) is active if **any** token pair within that block would be unmasked.
        """
        B = int(self.block_size)
        num_blocks = (int(seq_len) + B - 1) // B

        block_mask = torch.zeros((num_blocks, num_blocks), dtype=torch.bool)  # CPU

        # Token->chunk mapping is the only O(seq_len) step we keep.
        token_to_chunk = self._map_tokens_to_chunks(code, chunks, seq_len)

        # Precompute block lists for each chunk (CPU).
        num_chunks = len(chunks)
        chunk_blocks: List[List[int]] = [[] for _ in range(num_chunks)]
        seen = [set() for _ in range(num_chunks)]
        for t in range(int(seq_len)):
            c = token_to_chunk[t]
            if c < 0:
                continue
            b = t // B
            if b not in seen[c]:
                seen[c].add(b)
                chunk_blocks[c].append(b)

        # Rule 0: sliding window (union over tokens in the query block).
        w = int(self.window_size)
        for qb in range(num_blocks):
            q_start = qb * B
            k_min = max(0, q_start - w)
            j_min = k_min // B
            j_max = qb  # keys cannot exceed query block under causal
            if j_min <= j_max:
                block_mask[qb, j_min : j_max + 1] = True

        # Rule 1: within-chunk visibility (optionally limited by max_chunk_tokens)
        if self.max_chunk_tokens > 0:
            blocks_per_group = max(1, int(self.max_chunk_tokens) // B)
            for blocks in chunk_blocks:
                if len(blocks) <= 1:
                    if len(blocks) == 1:
                        b = blocks[0]
                        block_mask[b, b] = True
                    continue
                by_group: Dict[int, List[int]] = {}
                for b in blocks:
                    g = b // blocks_per_group
                    by_group.setdefault(g, []).append(b)
                for grp_blocks in by_group.values():
                    if not grp_blocks:
                        continue
                    bb = torch.tensor(grp_blocks, dtype=torch.long)
                    block_mask[bb[:, None], bb[None, :]] = True
        else:
            for blocks in chunk_blocks:
                if not blocks:
                    continue
                bb = torch.tensor(blocks, dtype=torch.long)
                block_mask[bb[:, None], bb[None, :]] = True

        # Rule 2: chunk dependencies (union to target chunk blocks)
        if chunk_deps:
            for src, tgts in chunk_deps.items():
                if not (0 <= int(src) < num_chunks):
                    continue
                src_blocks = chunk_blocks[int(src)]
                if not src_blocks:
                    continue
                tgt_blocks: List[int] = []
                for t in tgts:
                    if not (0 <= int(t) < num_chunks):
                        continue
                    tgt_blocks.extend(chunk_blocks[int(t)])
                if not tgt_blocks:
                    continue
                src_bb = torch.tensor(src_blocks, dtype=torch.long)
                tgt_bb = torch.tensor(sorted(set(tgt_blocks)), dtype=torch.long)
                block_mask[src_bb[:, None], tgt_bb[None, :]] = True

        # Rule 3: globally-visible keys (imports, signatures, specials, prefix/suffix, last-k chunks)
        # Import chunks: all queries can see their key blocks
        if import_chunks:
            key_blocks: List[int] = []
            for c in import_chunks:
                if 0 <= int(c) < num_chunks:
                    key_blocks.extend(chunk_blocks[int(c)])
            if key_blocks:
                kb = torch.tensor(sorted(set(key_blocks)), dtype=torch.long)
                block_mask[:, kb] = True

        # Function/class signatures + special lines (token-level selection, then reduce to blocks)
        globally_visible_lines = set()
        for chunk in chunks:
            if chunk.chunk_type in [ChunkType.FUNCTION, ChunkType.CLASS, ChunkType.METHOD]:
                if hasattr(chunk, "signature_start_line") and hasattr(chunk, "signature_end_line"):
                    if chunk.signature_start_line and chunk.signature_end_line:
                        globally_visible_lines.update(range(chunk.signature_start_line, chunk.signature_end_line + 1))

        special_lines = self._identify_special_lines(code, chunks, language=language)
        globally_visible_lines.update(special_lines["global_vars"])
        globally_visible_lines.update(special_lines["type_aliases"])
        globally_visible_lines.update(special_lines["class_attrs"])
        globally_visible_lines.update(special_lines["return_stmts"])

        if globally_visible_lines:
            global_token_mask = self._map_lines_to_tokens(code, globally_visible_lines, seq_len)
            idx = torch.nonzero(global_token_mask, as_tuple=True)[0]
            if idx.numel() > 0:
                kb = torch.unique((idx // B).to(torch.long))
                block_mask[:, kb] = True

        # Prefix / suffix globals
        num_prefix_global = min(int(self.num_prefix_tokens), int(seq_len))
        if num_prefix_global > 0:
            last = (num_prefix_global - 1) // B
            kb = torch.arange(last + 1, dtype=torch.long)
            block_mask[:, kb] = True

        num_suffix_global = min(int(self.num_suffix_tokens), int(seq_len))
        if num_suffix_global > 0:
            start = max(0, int(seq_len) - num_suffix_global)
            kb = torch.arange(start // B, num_blocks, dtype=torch.long)
            block_mask[:, kb] = True

        # Last-k chunks globally visible
        k = int(self.global_last_k_chunks)
        if k > 0 and num_chunks > 0:
            start = max(0, num_chunks - k)
            key_blocks = []
            for c in range(start, num_chunks):
                key_blocks.extend(chunk_blocks[c])
            if key_blocks:
                kb = torch.tensor(sorted(set(key_blocks)), dtype=torch.long)
                block_mask[:, kb] = True

        # Causal in block space
        block_mask = torch.tril(block_mask, diagonal=0)
        return block_mask.to(dtype=torch.int32)

    def build_block_indices(
        self,
        code: str,
        seq_len: int,
        device: torch.device,
        *,
        batch_size: int = 1,
        crossfile_code: Optional[str] = None,
        current_code: Optional[str] = None,
        language: Optional[str] = None,
        crossfile_full_attention: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return (block_indices, block_counts, block_mask_2d) for the given code/seq_len.
        This avoids materializing the token-level mask when block-level sparsity is used.
        """
        if crossfile_code is not None and current_code is not None:
            # Rebuild chunks/deps the same way as _build_mask_with_crossfile, but stop at chunk-level structures.
            if not code.startswith(crossfile_code):
                raise ValueError("full_code does not start with crossfile_code; dataset fields are inconsistent.")

            crossfile_num_lines = len(crossfile_code.split("\n")) if crossfile_code else 0
            chunks: List = []
            import_chunks: Set[int] = set()
            global_chunks: Set[int] = set()
            chunk_deps: Dict[int, Set[int]] = {}

            # Crossfile chunks
            if crossfile_code and crossfile_code.strip():
                if crossfile_full_attention:
                    chunks.append(
                        CodeChunk(
                            chunk_type=ChunkType.MODULE_CODE,
                            start=0,
                            end=len(crossfile_code),
                            text=crossfile_code,
                            name="",
                            parent_name=None,
                            start_line=1,
                            end_line=crossfile_num_lines,
                            signature_start_line=None,
                            signature_end_line=None,
                        )
                    )
                    global_chunks.add(len(chunks) - 1)
                else:
                    lines = crossfile_code.split("\n")
                    header_prefixes = ("# ", "// ")

                    def _is_header(s: str) -> bool:
                        s = s.rstrip()
                        if not s.startswith(header_prefixes):
                            return False
                        import re
                        return re.match(r"^(#|//)\s+.+:\s*\d+\s*-\s*\d+\s*$", s) is not None

                    headers = [i for i, ln in enumerate(lines, start=1) if _is_header(ln)]
                    boundaries: List[tuple[int, int]] = []
                    if headers:
                        if headers[0] > 1:
                            boundaries.append((1, headers[0] - 1))
                        for a, b in zip(headers, headers[1:] + [len(lines) + 1]):
                            boundaries.append((a, b - 1))
                    else:
                        boundaries.append((1, len(lines)))

                    def _all_blank(s: int, e: int) -> bool:
                        return all(not lines[i - 1].strip() for i in range(s, e + 1))

                    merged: List[tuple[int, int]] = []
                    pending_blank: Optional[tuple[int, int]] = None
                    for s, e in boundaries:
                        if e < s:
                            continue
                        if _all_blank(s, e):
                            if merged:
                                ps, pe = merged[-1]
                                merged[-1] = (ps, max(pe, e))
                            else:
                                pending_blank = (s, e) if pending_blank is None else (pending_blank[0], max(pending_blank[1], e))
                            continue
                        if pending_blank is not None:
                            s = min(s, pending_blank[0])
                            pending_blank = None
                        merged.append((s, e))

                    for start_line, end_line in merged:
                        seg_text = "\n".join(lines[start_line - 1 : end_line])
                        chunks.append(
                            CodeChunk(
                                chunk_type=ChunkType.MODULE_CODE,
                                start=0,
                                end=len(seg_text),
                                text=seg_text,
                                name="",
                                parent_name=None,
                                start_line=start_line,
                                end_line=end_line,
                                signature_start_line=None,
                                signature_end_line=None,
                            )
                        )

            crossfile_chunk_count = len(chunks)

            # Current chunks via tree-sitter
            if current_code and current_code.strip():
                segmenter = TreeSitterSegmenter(tokenizer=self.tokenizer)
                current_chunks = segmenter.segment_code(current_code, language=language)
                current_chunks_filled = self._append_uncovered_line_chunks(current_code, current_chunks)
                shifted_chunks = []
                from dataclasses import replace as _replace
                for c in current_chunks_filled:
                    shifted_chunks.append(
                        _replace(
                            c,
                            start_line=c.start_line + crossfile_num_lines,
                            end_line=c.end_line + crossfile_num_lines,
                            signature_start_line=(c.signature_start_line + crossfile_num_lines) if c.signature_start_line else None,
                            signature_end_line=(c.signature_end_line + crossfile_num_lines) if c.signature_end_line else None,
                        )
                    )
                chunks.extend(shifted_chunks)

                analyzer = CrossReferenceAnalyzer(current_chunks, current_code, language=language)
                references = analyzer.analyze()
                for ref in references:
                    src = ref.source_chunk_idx + crossfile_chunk_count
                    tgt = ref.target_chunk_idx + crossfile_chunk_count
                    chunk_deps.setdefault(src, set()).add(tgt)
                    chunk_deps.setdefault(tgt, set()).add(src)

                current_imports = {i for i, c in enumerate(current_chunks) if c.chunk_type == ChunkType.IMPORT}
                import_chunks = {i + crossfile_chunk_count for i in current_imports}

            # Similarity edges
            if self.use_chunk_similarity:
                src_idx = list(range(crossfile_chunk_count, len(chunks)))
                tgt_idx = list(range(0, crossfile_chunk_count))
                sim_deps = self._similarity_edges(chunks, code, src_indices=src_idx, tgt_indices=tgt_idx)
                for src, tgts in sim_deps.items():
                    chunk_deps.setdefault(src, set()).update(tgts)

                if self.use_crossfile_chunk_similarity and crossfile_chunk_count > 1:
                    cf_idx = list(range(0, crossfile_chunk_count))
                    sim_cf = self._similarity_edges(
                        chunks,
                        code,
                        src_indices=cf_idx,
                        tgt_indices=cf_idx,
                        top_percent=self.crossfile_chunk_similarity_top_percent,
                    )
                    for src, tgts in sim_cf.items():
                        chunk_deps.setdefault(src, set()).update(tgts)

            merged_import_chunks = set(import_chunks)
            if global_chunks:
                merged_import_chunks.update(global_chunks)

            block_mask_2d = self._build_block_mask_from_chunks(
                code=code,
                seq_len=seq_len,
                chunks=chunks,
                chunk_deps=chunk_deps,
                import_chunks=merged_import_chunks,
                language=language,
            )
        else:
            code_hash = hash(code)
            lang_key = (str(language).lower() if language else None)
            chunk_cache_key = (code_hash, lang_key)
            global _GLOBAL_CHUNK_CACHE
            if chunk_cache_key not in _GLOBAL_CHUNK_CACHE:
                segmenter = TreeSitterSegmenter(tokenizer=self.tokenizer)
                chunks = segmenter.segment_code(code, language=language)
                import_chunks = {i for i, c in enumerate(chunks) if c.chunk_type == ChunkType.IMPORT}

                chunk_deps: Dict[int, Set[int]] = {}
                analyzer = CrossReferenceAnalyzer(chunks, code, language=language)
                references = analyzer.analyze()
                for ref in references:
                    src, tgt = ref.source_chunk_idx, ref.target_chunk_idx
                    chunk_deps.setdefault(src, set()).add(tgt)
                    chunk_deps.setdefault(tgt, set()).add(src)

                chunks = self._append_uncovered_line_chunks(code, chunks)
                _GLOBAL_CHUNK_CACHE[chunk_cache_key] = {"chunks": chunks, "chunk_deps": chunk_deps, "import_chunks": import_chunks}

            cached = _GLOBAL_CHUNK_CACHE[chunk_cache_key]
            chunks = cached["chunks"]
            chunk_deps = cached["chunk_deps"]
            import_chunks = cached["import_chunks"]

            if self.use_chunk_similarity:
                sim_deps = self._similarity_edges(chunks, code)
                if sim_deps:
                    merged = {k: set(v) for k, v in chunk_deps.items()} if chunk_deps else {}
                    for src, tgts in sim_deps.items():
                        merged.setdefault(src, set()).update(tgts)
                    chunk_deps = merged

            block_mask_2d = self._build_block_mask_from_chunks(
                code=code,
                seq_len=seq_len,
                chunks=chunks,
                chunk_deps=chunk_deps,
                import_chunks=import_chunks,
                language=language,
            )

        # Compute indices on CPU (small) then move to target device.
        block_mask_expanded = block_mask_2d.unsqueeze(0).expand(int(batch_size), -1, -1).contiguous()
        block_indices, block_counts = compute_block_indices_from_mask(block_mask_expanded)
        return block_indices.to(device), block_counts.to(device), block_mask_2d.to(device)

    def clear_cache(self):
        """Internal utility function."""
        global _GLOBAL_CHUNK_CACHE, _GLOBAL_LINE_TO_TOKEN_CACHE, _GLOBAL_MASK_CACHE, _GLOBAL_SIM_EDGES_CACHE, _GLOBAL_SIM_VEC_CACHE
        _GLOBAL_CHUNK_CACHE.clear()
        _GLOBAL_LINE_TO_TOKEN_CACHE.clear()
        _GLOBAL_MASK_CACHE.clear()
        _GLOBAL_SIM_EDGES_CACHE.clear()
        _GLOBAL_SIM_VEC_CACHE.clear()


# ==================== Block-Level Mask Computation ====================

_BLOCK_MASK_CACHE = {}
_BLOCK_MASK_CACHE_MAX_SIZE = 50


def compute_block_indices_from_mask(block_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Internal utility function."""
    batch, num_blocks_m, num_blocks_n = block_mask.shape
    device = block_mask.device

    block_counts = block_mask.sum(dim=2)  # (batch, num_blocks_M)
    max_active_blocks = int(block_counts.max().item())

    if max_active_blocks == 0:
        max_active_blocks = 1

    block_mask_float = block_mask.float()

    sorted_indices = torch.argsort(block_mask_float, dim=2, descending=True)

    block_indices = sorted_indices[:, :, :max_active_blocks]

    range_tensor = torch.arange(max_active_blocks, device=device)
    valid_mask = range_tensor.unsqueeze(0).unsqueeze(0) < block_counts.unsqueeze(2)

    block_indices = torch.where(valid_mask, block_indices, torch.tensor(-1, dtype=torch.int32, device=device))

    return block_indices.to(torch.int32), block_counts.to(torch.int32)


def compute_block_mask_from_token_mask(
    token_mask: torch.Tensor,
    block_size_m: int,
    block_size_n: int
) -> torch.Tensor:
    """Internal utility function."""
    seq_len = token_mask.shape[0]
    num_blocks_m = (seq_len + block_size_m - 1) // block_size_m
    num_blocks_n = (seq_len + block_size_n - 1) // block_size_n

    pad_m = num_blocks_m * block_size_m - seq_len
    pad_n = num_blocks_n * block_size_n - seq_len

    if pad_m > 0 or pad_n > 0:
        token_mask = torch.nn.functional.pad(token_mask, (0, pad_n, 0, pad_m), value=float('-inf'))

    # Reshape to blocks: (num_blocks_m, block_size_m, num_blocks_n, block_size_n)
    token_mask = token_mask.view(num_blocks_m, block_size_m, num_blocks_n, block_size_n)

    # Check if any element in each block is not -inf
    # (num_blocks_m, num_blocks_n)
    block_mask = (~torch.isinf(token_mask)).any(dim=(1, 3)).to(torch.int32)

    return block_mask


# ==================== Triton Sparse Attention ====================

class StructureAwareSparseAttentionTriton(nn.Module):
    """Internal utility function."""

    def __init__(
        self,
        tokenizer,
        block_size=64,
        window_size=64,
        num_prefix_tokens=128,
        num_suffix_tokens=256,
        use_token_level_sparsity=False,
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
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.window_size = window_size
        self.num_prefix_tokens = num_prefix_tokens
        self.num_suffix_tokens = num_suffix_tokens
        self.use_token_level_sparsity = use_token_level_sparsity
        self.mask_builder = StructureAwareMaskBuilder(
            tokenizer=tokenizer,
            block_size=self.block_size,
            window_size=window_size,
            num_prefix_tokens=num_prefix_tokens,
            num_suffix_tokens=num_suffix_tokens,
            global_last_k_chunks=global_last_k_chunks,
            max_chunk_tokens=max_chunk_tokens,
            embedding_weight=embedding_weight,
            use_chunk_similarity=use_chunk_similarity,
            chunk_similarity_top_percent=chunk_similarity_top_percent,
            chunk_similarity_max_tokens_per_chunk=chunk_similarity_max_tokens_per_chunk,
            chunk_similarity_max_neighbors=chunk_similarity_max_neighbors,
            use_crossfile_chunk_similarity=use_crossfile_chunk_similarity,
            crossfile_chunk_similarity_top_percent=crossfile_chunk_similarity_top_percent,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        code: str = None,
        attention_mask: Optional[torch.Tensor] = None,
        precomputed_block_indices: Optional[torch.Tensor] = None,
        precomputed_block_counts: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Internal utility function."""
        batch_size, num_heads, query_len, head_dim = query.shape
        key_len = key.shape[2]

        if code is not None and query_len == key_len and query_len > 1:
            BLOCK_M = self.block_size
            BLOCK_N = self.block_size
            
            token_mask = None
            
            if precomputed_block_indices is not None and precomputed_block_counts is not None:
                block_indices = precomputed_block_indices
                block_counts = precomputed_block_counts
            else:
                from sabrecoder_llama_triton import get_precomputed_blocks
                global_block_indices, global_block_counts = get_precomputed_blocks()

                if global_block_indices is not None and global_block_counts is not None:
                    block_indices = global_block_indices
                    block_counts = global_block_counts
                else:
                    global _BLOCK_MASK_CACHE, _BLOCK_MASK_CACHE_MAX_SIZE
                    code_hash = hash(code)
                    # Include mask-builder config to avoid reusing indices across different sparsity settings.
                    block_cache_key = (
                        code_hash,
                        query_len,
                        BLOCK_M,
                        BLOCK_N,
                        query.device,
                        batch_size,
                        self.mask_builder.window_size,
                        self.mask_builder.num_prefix_tokens,
                        self.mask_builder.num_suffix_tokens,
                        self.mask_builder.block_size,
                        self.mask_builder.global_last_k_chunks,
                        self.mask_builder.max_chunk_tokens,
                        self.mask_builder.use_chunk_similarity,
                        self.mask_builder.chunk_similarity_top_percent,
                        self.mask_builder.chunk_similarity_max_tokens_per_chunk,
                        self.mask_builder.chunk_similarity_max_neighbors,
                        self.mask_builder.use_crossfile_chunk_similarity,
                        self.mask_builder.crossfile_chunk_similarity_top_percent,
                    )

                    if block_cache_key in _BLOCK_MASK_CACHE:
                        block_indices, block_counts = _BLOCK_MASK_CACHE[block_cache_key]
                    else:
                        if self.use_token_level_sparsity:
                            token_mask = self.mask_builder.build_token_level_mask(
                                code, query_len, query.device, query.dtype
                            )
                            block_mask = compute_block_mask_from_token_mask(token_mask, BLOCK_M, BLOCK_N)
                            block_mask = block_mask.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
                            block_indices, block_counts = compute_block_indices_from_mask(block_mask)
                        else:
                            block_indices, block_counts, _ = self.mask_builder.build_block_indices(
                                code, query_len, query.device, batch_size=batch_size
                            )

                        if len(_BLOCK_MASK_CACHE) >= _BLOCK_MASK_CACHE_MAX_SIZE:
                            _BLOCK_MASK_CACHE.pop(next(iter(_BLOCK_MASK_CACHE)))
                        _BLOCK_MASK_CACHE[block_cache_key] = (block_indices.clone(), block_counts.clone())

            num_blocks_m = (query_len + BLOCK_M - 1) // BLOCK_M
            max_active = block_indices.shape[2]  # max_active_blocks

            output = torch.empty_like(query)
            grid = (num_blocks_m, batch_size * num_heads)

            if self.use_token_level_sparsity:
                if token_mask is None:
                    token_mask = self.mask_builder.build_token_level_mask(
                        code, query_len, query.device, query.dtype
                    )
                
                _fwd_kernel_sabrecoder_token_sparse[grid](
                    query, key, value, output,
                    block_indices, block_counts,
                    token_mask,
                    query.stride(0), query.stride(1), query.stride(2), query.stride(3),
                    key.stride(0), key.stride(1), key.stride(2), key.stride(3),
                    value.stride(0), value.stride(1), value.stride(2), value.stride(3),
                    output.stride(0), output.stride(1), output.stride(2), output.stride(3),
                    block_indices.stride(0), block_indices.stride(1), block_indices.stride(2),
                    block_counts.stride(0), block_counts.stride(1),
                    token_mask.stride(0), token_mask.stride(1),
                    batch_size, num_heads, query_len,
                    max_active_blocks=max_active,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_DMODEL=head_dim,
                )
            else:
                _fwd_kernel_sabrecoder[grid](
                    query, key, value, output,
                    block_indices, block_counts,
                    query.stride(0), query.stride(1), query.stride(2), query.stride(3),
                    key.stride(0), key.stride(1), key.stride(2), key.stride(3),
                    value.stride(0), value.stride(1), value.stride(2), value.stride(3),
                    output.stride(0), output.stride(1), output.stride(2), output.stride(3),
                    block_indices.stride(0), block_indices.stride(1), block_indices.stride(2),
                    block_counts.stride(0), block_counts.stride(1),
                    batch_size, num_heads, query_len,
                    max_active_blocks=max_active,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_DMODEL=head_dim,
                )

            return output
        else:
            if attention_mask is not None:
                if attention_mask.shape[-2:] != (query_len, key_len):
                    attention_mask = None

            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=True if attention_mask is None else False,
            )

    def clear_cache(self):
        """Clear cached mask builder state."""
        self.mask_builder.clear_cache()
        global _BLOCK_MASK_CACHE
        _BLOCK_MASK_CACHE.clear()


def get_cache_stats():
    """Return cache sizes used by SabreCoder attention."""
    global _GLOBAL_CHUNK_CACHE, _GLOBAL_LINE_TO_TOKEN_CACHE, _GLOBAL_MASK_CACHE, _BLOCK_MASK_CACHE, _GLOBAL_SIM_EDGES_CACHE, _GLOBAL_SIM_VEC_CACHE

    stats = {
        'chunk_cache_size': len(_GLOBAL_CHUNK_CACHE),
        'line_to_token_cache_size': len(_GLOBAL_LINE_TO_TOKEN_CACHE),
        'mask_cache_size': len(_GLOBAL_MASK_CACHE),
        'block_mask_cache_size': len(_BLOCK_MASK_CACHE),
        'sim_edges_cache_size': len(_GLOBAL_SIM_EDGES_CACHE),
        'sim_vec_cache_size': len(_GLOBAL_SIM_VEC_CACHE),
        'total_cached_items': (
            len(_GLOBAL_CHUNK_CACHE) +
            len(_GLOBAL_LINE_TO_TOKEN_CACHE) +
            len(_GLOBAL_MASK_CACHE) +
            len(_BLOCK_MASK_CACHE) +
            len(_GLOBAL_SIM_EDGES_CACHE) +
            len(_GLOBAL_SIM_VEC_CACHE)
        )
    }
    return stats


def print_cache_stats():
    """Print cache sizes used by SabreCoder attention."""
    stats = get_cache_stats()
    print("\n" + "="*60)
    print("CACHE STATISTICS")
    print("="*60)
    print(f"  Chunk cache (code structure):     {stats['chunk_cache_size']:3d} items")
    print(f"  Line-to-token mapping cache:      {stats['line_to_token_cache_size']:3d} items")
    print(f"  Token-level mask cache:           {stats['mask_cache_size']:3d} items")
    print(f"  Block-level mask cache:           {stats['block_mask_cache_size']:3d} items")
    print(f"  Similarity edges cache:           {stats['sim_edges_cache_size']:3d} items")
    print(f"  Similarity vectors cache:         {stats['sim_vec_cache_size']:3d} items")
    print(f"  Total cached items:               {stats['total_cached_items']:3d} items")
    print("="*60 + "\n")


def clear_all_caches():
    """Clear all SabreCoder attention caches."""
    global _GLOBAL_CHUNK_CACHE, _GLOBAL_LINE_TO_TOKEN_CACHE, _GLOBAL_MASK_CACHE, _BLOCK_MASK_CACHE, _GLOBAL_SIM_EDGES_CACHE, _GLOBAL_SIM_VEC_CACHE
    _GLOBAL_CHUNK_CACHE.clear()
    _GLOBAL_LINE_TO_TOKEN_CACHE.clear()
    _GLOBAL_MASK_CACHE.clear()
    _BLOCK_MASK_CACHE.clear()
    _GLOBAL_SIM_EDGES_CACHE.clear()
    _GLOBAL_SIM_VEC_CACHE.clear()
