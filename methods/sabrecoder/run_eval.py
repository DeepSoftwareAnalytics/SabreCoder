"""End-to-end SabreCoder evaluation entrypoint."""

import argparse
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

# Add evaluation utilities to path
sys.path.insert(0, PROJECT_ROOT)

# Import local modules
from sabrecoder_llama_triton import (
    patch_model_with_sabrecoder_attention_triton,
    get_attention_call_counts,
    get_sparsity_stats,
    set_current_code,
    clear_current_code,
    enable_attention_timing,
    get_attention_timing,
    reset_attention_timing,
    set_precomputed_blocks,
    clear_precomputed_blocks
)

from sabrecoder_attention_triton import (
    print_cache_stats,
)

from evaluation import (
    load_lcc_dataset,
    detect_dataset_type,
    warmup_model,
    get_device_info
)
from evaluation.save_results_unified import save_unified_results


def _resolve_input_path(path: str | None) -> str | None:
    if path is None or os.path.isabs(path):
        return path

    cwd_path = os.path.abspath(path)
    repo_path = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(cwd_path) or not os.path.exists(repo_path):
        return cwd_path
    return repo_path


def _resolve_output_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


class SabreCoderEvaluatorTriton:
    """Evaluator for SabreCoder with Triton sparse attention."""

    def __init__(
        self,
        model,
        tokenizer,
        device,
        max_length=4096,
        max_new_tokens=64,
        use_truncation=True,
        verbose=False,
        block_size: int = 64,
        window_size=64,
        num_prefix_tokens=128,
        num_suffix_tokens=256,
        global_last_k_chunks: int = 2,
        max_chunk_tokens: int = 0,
        precomputed_blocks=None,
        skip_precompute=False,
        dataset_type='lcc',
        language=None,
        crossfile_full_attention: bool = True,
        embedding_weight=None,
        use_chunk_similarity: bool = False,
        chunk_similarity_top_percent: float = 0.1,
        chunk_similarity_max_tokens_per_chunk: int = 256,
        chunk_similarity_max_neighbors: int = 8,
        use_crossfile_chunk_similarity: bool = False,
        crossfile_chunk_similarity_top_percent=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.use_truncation = use_truncation
        self.verbose = verbose
        self.block_size = int(block_size)
        self.window_size = window_size
        self.num_prefix_tokens = num_prefix_tokens
        self.num_suffix_tokens = num_suffix_tokens
        self.global_last_k_chunks = int(global_last_k_chunks)
        self.max_chunk_tokens = int(max_chunk_tokens)
        self.skip_precompute = skip_precompute
        self.dataset_type = dataset_type.lower()
        self.language = language
        self.crossfile_full_attention = crossfile_full_attention
        self.embedding_weight = embedding_weight
        self.use_chunk_similarity = bool(use_chunk_similarity)
        self.chunk_similarity_top_percent = float(chunk_similarity_top_percent)
        self.chunk_similarity_max_tokens_per_chunk = int(chunk_similarity_max_tokens_per_chunk)
        self.chunk_similarity_max_neighbors = int(chunk_similarity_max_neighbors)
        self.use_crossfile_chunk_similarity = bool(use_crossfile_chunk_similarity)
        self.crossfile_chunk_similarity_top_percent = crossfile_chunk_similarity_top_percent

        from sabrecoder_attention_triton import StructureAwareMaskBuilder
        self.mask_builder = StructureAwareMaskBuilder(
            tokenizer=tokenizer,
            block_size=self.block_size,
            window_size=window_size,
            num_prefix_tokens=num_prefix_tokens,
            num_suffix_tokens=num_suffix_tokens,
            global_last_k_chunks=self.global_last_k_chunks,
            max_chunk_tokens=self.max_chunk_tokens,
            embedding_weight=self.embedding_weight,
            use_chunk_similarity=self.use_chunk_similarity,
            chunk_similarity_top_percent=self.chunk_similarity_top_percent,
            chunk_similarity_max_tokens_per_chunk=self.chunk_similarity_max_tokens_per_chunk,
            chunk_similarity_max_neighbors=self.chunk_similarity_max_neighbors,
            use_crossfile_chunk_similarity=self.use_crossfile_chunk_similarity,
            crossfile_chunk_similarity_top_percent=self.crossfile_chunk_similarity_top_percent,
        )
        if verbose:
            print(f"Tokenizer truncation_side: {self.tokenizer.truncation_side}")

        self.precomputed_blocks = precomputed_blocks or {}

    def evaluate(self, dataset):
        """Run evaluation on a dataset."""
        import time
        from tqdm import tqdm
        from evaluation.metrics import (
            compute_edit_similarity, compute_f1, compute_exact_match,
            compute_cceval_edit_similarity, compute_cceval_identifier_f1, compute_cceval_exact_match,
            truncate_to_answer_length
        )
        from evaluation.cceval_utils import postprocess_code_lines

        from sabrecoder_llama_triton import enable_sparsity_tracking, update_sparsity_stats
        enable_sparsity_tracking(True)

        if not getattr(self, "_sparse_prefill_warmup_done", False):
            try:
                from sabrecoder_llama_triton import set_precomputed_blocks, clear_precomputed_blocks

                warm_code = "def _warmup_sparse_prefill():\n    return 0\n"
                warm_inputs = self.tokenizer(
                    warm_code,
                    return_tensors='pt',
                    truncation=True,
                    max_length=min(int(self.max_length), 256),
                    add_special_tokens=True,
                ).to(self.device)

                warm_len = int(warm_inputs["input_ids"].shape[1])
                bi, bc, _ = self.mask_builder.build_block_indices(
                    warm_code,
                    warm_len,
                    self.device,
                    batch_size=1,
                    language=self.language,
                    crossfile_full_attention=self.crossfile_full_attention,
                )

                set_precomputed_blocks(bi, bc)
                with torch.no_grad():
                    _ = self.model(
                        input_ids=warm_inputs["input_ids"],
                        attention_mask=warm_inputs.get("attention_mask", None),
                        return_dict=True,
                    )
            finally:
                try:
                    clear_precomputed_blocks()
                except Exception:
                    pass
            self._sparse_prefill_warmup_done = True

        if self.skip_precompute and self.precomputed_blocks:
            print(f"\nUsing provided precomputed blocks ({len(self.precomputed_blocks)} samples), skip precompute.\n")
            from sabrecoder_llama_triton import update_sparsity_stats_from_blocks

            from sabrecoder_llama_triton import enable_sparsity_tracking
            enable_sparsity_tracking(True)

            for _, (block_indices, block_counts) in self.precomputed_blocks.items():
                if block_counts.dim() == 2:
                    num_blocks_m = int(block_counts.shape[1])
                else:
                    num_blocks_m = int(block_counts.shape[0])

                total_causal_blocks = num_blocks_m * (num_blocks_m + 1) // 2
                active_blocks = int(block_counts.sum().item())
                update_sparsity_stats_from_blocks(total_causal_blocks, active_blocks)
        else:
            print("\nPrecomputing masks and block indices for all samples...")

            for idx, sample in enumerate(tqdm(dataset, desc="Pre-computing")):
                code = sample.get('context', sample.get('input', ''))

                crossfile_code = sample.get('crossfile_context', None)
                current_code = sample.get('current_file_context', None)

                inputs = self.tokenizer(
                    code,
                    return_tensors='pt',
                    truncation=self.use_truncation,
                    max_length=self.max_length,
                    add_special_tokens=True
                ).to(self.device)

                input_len = inputs['input_ids'].shape[1]

                block_indices, block_counts, block_mask = self.mask_builder.build_block_indices(
                    code,
                    input_len,
                    self.device,
                    batch_size=1,
                    crossfile_code=crossfile_code,
                    current_code=current_code,
                    language=self.language,
                    crossfile_full_attention=self.crossfile_full_attention,
                )

                update_sparsity_stats(block_mask)

                self.precomputed_blocks[idx] = (block_indices.clone(), block_counts.clone())

                del inputs, block_mask

            print(f"Finished precomputing masks and block indices. Stored {len(self.precomputed_blocks)} samples.\n")

        results = {
            'num_samples': len(dataset),
            'sample_details': [],
            'avg_prefill_time': 0,
            'avg_prefill_attn_time': 0,
            'avg_generation_time': 0,
            'avg_ttft_ms': 0,
            'avg_input_ppl': 0,
            'avg_generation_ppl': 0,
            'avg_exact_match': 0,
            'avg_f1': 0,
            'avg_edit_similarity': 0,
            'avg_memory_mb': 0,
            'prefill_throughput': 0,
            'generation_throughput': 0,
        }

        total_prefill = 0
        total_prefill_attn = 0
        total_generation = 0
        total_ttft = 0
        total_input_ppl = 0
        total_gen_ppl = 0
        total_em = 0
        total_f1 = 0
        total_es = 0
        total_memory = 0
        total_input_tokens = 0
        total_output_tokens = 0

        for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
            code = sample.get('context', sample.get('input', ''))
            ground_truth = sample.get('answers', sample.get('ground_truth', ''))
            
            crossfile_code = sample.get('crossfile_context', None)
            current_code = sample.get('current_file_context', None)

            set_current_code(code)

            try:
                inputs = self.tokenizer(
                    code,
                    return_tensors='pt',
                    truncation=self.use_truncation,
                    max_length=self.max_length,
                    add_special_tokens=True
                ).to(self.device)

                input_ids = inputs['input_ids']
                input_len = input_ids.shape[1]
                
                num_blocks = (input_len + self.block_size - 1) // self.block_size
                pre = self.precomputed_blocks.get(idx, None)
                if pre is not None:
                    block_indices, block_counts = pre
                    ok = (block_counts.dim() == 2 and int(block_counts.shape[1]) == int(num_blocks))
                else:
                    ok = False

                if not ok:
                    block_indices, block_counts, _ = self.mask_builder.build_block_indices(
                        code,
                        input_len,
                        self.device,
                        batch_size=1,
                        crossfile_code=crossfile_code,
                        current_code=current_code,
                        language=self.language,
                        crossfile_full_attention=self.crossfile_full_attention,
                    )

                set_precomputed_blocks(block_indices, block_counts)

                enable_attention_timing(True)
                reset_attention_timing()

                device_type = self.device if isinstance(self.device, str) else self.device.type
                torch.cuda.synchronize() if device_type == 'cuda' else None
                prefill_start = time.time()

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=inputs['attention_mask'],
                        return_dict=True
                    )

                torch.cuda.synchronize() if device_type == 'cuda' else None
                prefill_time = time.time() - prefill_start

                attn_stats = get_attention_timing()
                prefill_attn_time = attn_stats['total_time']

                enable_attention_timing(False)

                torch.cuda.synchronize() if device_type == 'cuda' else None
                gen_start = time.time()

                with torch.no_grad():
                    gen_outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )

                torch.cuda.synchronize() if device_type == 'cuda' else None
                gen_time = time.time() - gen_start

                generated_ids = gen_outputs[0, input_len:]
                output_len = len(generated_ids)

                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

                enable_attention_timing(False)

                if output_len > 0:
                    with torch.no_grad():
                        full_sequence = gen_outputs

                        labels = full_sequence.clone()
                        labels[:, :input_len] = -100

                        clear_current_code()

                        gen_outputs_for_loss = self.model(
                            input_ids=full_sequence,
                            labels=labels,
                            return_dict=True
                        )

                        set_current_code(code)

                        gen_loss = gen_outputs_for_loss.loss
                        gen_ppl = torch.exp(gen_loss).item() if gen_loss is not None else 0.0
                else:
                    gen_ppl = 0.0

                ttft = prefill_time * 1000

                gt_text = ground_truth[0] if isinstance(ground_truth, list) else ground_truth

                if self.dataset_type == 'cceval':
                    truncated_text = postprocess_code_lines(
                        code,
                        generated_text,
                        self.language or 'python',
                        use_ast=False,
                    )
                else:
                    truncated_text = truncate_to_answer_length(
                        generated_text,
                        [gt_text],
                        self.tokenizer
                    )

                if self.dataset_type == 'cceval':
                    em = compute_cceval_exact_match(truncated_text, [gt_text], self.language)
                    f1 = compute_cceval_identifier_f1(truncated_text, [gt_text], self.language)
                    es = compute_cceval_edit_similarity(truncated_text, [gt_text]) / 100.0
                else:
                    em = compute_exact_match(truncated_text, [gt_text])
                    f1 = compute_f1(truncated_text, [gt_text])
                    es = compute_edit_similarity(truncated_text, [gt_text])

                if device_type == 'cuda':
                    memory_mb = torch.cuda.max_memory_allocated(device_type) / 1024 / 1024
                    torch.cuda.reset_peak_memory_stats(self.device)
                    torch.cuda.empty_cache()
                else:
                    memory_mb = 0

                sample_detail = {
                    'sample_idx': idx,
                    'context': code,
                    'crossfile_context': crossfile_code,
                    'current_file_context': current_code,
                    'input_length': input_len,
                    'output_length': output_len,
                    'prefill_time': prefill_time,
                    'prefill_attn_time': prefill_attn_time,
                    'generation_time': gen_time,
                    'ttft_ms': ttft,
                    'input_ppl': 0.0,
                    'generation_ppl': gen_ppl,
                    'em': em,
                    'f1': f1,
                    'es': es,
                    'memory_mb': memory_mb,
                    'generated_text': generated_text,
                    'truncated_text': truncated_text,
                    'ground_truth': gt_text
                }

                results['sample_details'].append(sample_detail)

                total_prefill += prefill_time
                total_prefill_attn += prefill_attn_time
                total_generation += gen_time
                total_ttft += ttft
                total_input_ppl += 0.0
                total_gen_ppl += gen_ppl
                total_em += em
                total_f1 += f1
                total_es += es
                total_memory += memory_mb
                total_input_tokens += input_len
                total_output_tokens += output_len

                if self.verbose and (idx + 1) % 5 == 0:
                    print(f"  Sample {idx+1}: Prefill={prefill_time:.3f}s, Gen={gen_time:.3f}s, EM={em:.2f}")

            except Exception as e:
                raise RuntimeError(f"Evaluation failed at sample_idx={idx}") from e
            finally:
                clear_current_code()

        n = len(dataset)
        results['avg_prefill_time'] = total_prefill / n
        results['avg_prefill_attn_time'] = total_prefill_attn / n
        results['avg_generation_time'] = total_generation / n
        results['avg_ttft_ms'] = total_ttft / n
        results['avg_input_ppl'] = total_input_ppl / n
        results['avg_generation_ppl'] = total_gen_ppl / n
        results['avg_exact_match'] = (total_em / n) * 100
        results['avg_f1'] = (total_f1 / n) * 100
        results['avg_edit_similarity'] = (total_es / n) * 100
        results['avg_memory_mb'] = total_memory / n

        results['prefill_throughput'] = total_input_tokens / total_prefill if total_prefill > 0 else 0
        results['generation_throughput'] = total_output_tokens / total_generation if total_generation > 0 else 0

        return results


def main():
    parser = argparse.ArgumentParser(description='SabreCoder evaluation')
    parser.add_argument('--model_name', type=str, default='deepseek-ai/deepseek-coder-1.3b-base',
                       help='HuggingFace model name')
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/_lcc_budget_prompts/LCC_python_test_ctx_12288_14336.jsonl',
        help='Path to a prepared evaluation dataset',
    )
    parser.add_argument('--max_samples', type=int, default=10,
                       help='Maximum number of samples to evaluate')
    parser.add_argument('--max_length', type=int, default=4096,
                       help='Maximum input length')
    parser.add_argument('--max_new_tokens', type=int, default=64,
                       help='Maximum tokens to generate')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--no_warmup', action='store_true',
                       help='Skip warmup runs')
    parser.add_argument('--use_sparse_for_generation', action='store_true',
                       help='Use sparse attention for generation (default: False, hybrid mode)')
    parser.add_argument('--use_compile', action='store_true',
                       help='Use torch.compile() for optimization (PyTorch 2.0+)')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead',
                       choices=['default', 'reduce-overhead', 'max-autotune'],
                       help='torch.compile mode (default: reduce-overhead)')
    parser.add_argument('--window_size', type=int, default=0,
                       help='Window size for sparse attention (default: 0)')
    parser.add_argument('--block_size', type=int, default=64,
                       help='Block/kernel size for Triton sparse attention (default: 64)')
    parser.add_argument('--num_prefix_tokens', type=int, default=4,
                       help='Number of prefix tokens to keep globally visible (default: 4)')
    parser.add_argument('--num_suffix_tokens', type=int, default=128,
                       help='Number of suffix tokens to keep globally visible (default: 128)')
    parser.add_argument('--global_last_k_chunks', type=int, default=0,
                       help='Make last K chunks globally visible (0 disables) (default: 0)')
    parser.add_argument('--max_chunk_tokens', type=int, default=128,
                       help='Max tokens per chunk for within-chunk visibility (block-aligned; 0 means unlimited). '
                            'Must be a multiple of --block_size.')
    parser.add_argument(
        '--crossfile_full_attention',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='For CCEval prompts, whether crossfile_context uses dense attention. '
             'If disabled, each retrieved segment is treated as a chunk.',
    )
    parser.add_argument('--use_chunk_similarity', action=argparse.BooleanOptionalAction, default=True,
                       help='Add similarity edges between chunks using token embedding lookup.')
    parser.add_argument('--chunk_similarity_top_percent', type=float, default=0.2,
                       help='Per-source neighbor ratio for chunk similarity edges. '
                            'k = floor(ratio * num_targets), capped by --chunk_similarity_max_neighbors; '
                            'k may be 0. Accepts fraction [0,1] or percent values greater than 1.')
    parser.add_argument('--chunk_similarity_max_tokens_per_chunk', type=int, default=128,
                       help='Max tokens per chunk when computing similarity vectors.')
    parser.add_argument('--chunk_similarity_max_neighbors', type=int, default=8,
                       help='Cap per-chunk similarity neighbors to keep sparsity meaningful.')
    parser.add_argument('--use_crossfile_chunk_similarity', action=argparse.BooleanOptionalAction, default=True,
                       help='Also add similarity edges within cross-file chunks. Requires --use_chunk_similarity.')
    parser.add_argument('--crossfile_chunk_similarity_top_percent', type=float, default=0.2,
                       help='Per-source neighbor ratio for crossfile chunk similarity edges. '
                            'Accepts fraction [0,1] or percent values greater than 1.')
    parser.add_argument('--use_block_level_mask', type=lambda x: x.lower() == 'true', default=True,
                       help='Use block-level mask (True) or token-level mask (False) (default: True)')
    parser.add_argument('--use_token_level_sparsity', action='store_true',
                       help='Apply token-level sparsity within active blocks.')
    parser.add_argument('--precomputed_blocks_path', type=str, default=None,
                       help='Path to precomputed block indices (.pt) to skip precompute')

    args = parser.parse_args()
    args.data_path = _resolve_input_path(args.data_path)
    args.output_dir = _resolve_output_path(args.output_dir)
    args.precomputed_blocks_path = _resolve_input_path(args.precomputed_blocks_path)
    if args.use_crossfile_chunk_similarity and not args.use_chunk_similarity:
        raise ValueError("--use_crossfile_chunk_similarity requires --use_chunk_similarity.")

    device = get_device_info()
    print(f"\n{'='*60}")
    print("SABRECODER EVALUATION")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.data_path}")
    print(f"Max samples: {args.max_samples}")
    print(f"Max input length: {args.max_length}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Window size: {args.window_size}")
    print(f"Block size: {args.block_size}")
    print(f"Prefix tokens: {args.num_prefix_tokens}")
    print(f"Suffix tokens: {args.num_suffix_tokens}")
    print(f"Global last-k chunks: {args.global_last_k_chunks}")
    print(f"Max chunk tokens (block-aligned): {args.max_chunk_tokens}")
    print(f"Implementation: Triton (True Block-Sparse)")
    mode = "Full Sparse" if args.use_sparse_for_generation else "Hybrid (Sparse Prefill, Dense Generation)"
    print(f"Mode: {mode}")
    sparsity_mode = "Token-level" if args.use_token_level_sparsity else "Block-level"
    print(f"Sparsity mode: {sparsity_mode}")
    print(f"{'='*60}\n")

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.truncation_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="eager"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

    print("\nPatching model with SabreCoder attention...")
    model = patch_model_with_sabrecoder_attention_triton(
        model,
        tokenizer=tokenizer,
        use_sparse_for_generation=args.use_sparse_for_generation,
        block_size=args.block_size,
        window_size=args.window_size,
        num_prefix_tokens=args.num_prefix_tokens,
        num_suffix_tokens=args.num_suffix_tokens,
        use_token_level_sparsity=args.use_token_level_sparsity,
        global_last_k_chunks=args.global_last_k_chunks,
        max_chunk_tokens=args.max_chunk_tokens,
        use_chunk_similarity=args.use_chunk_similarity,
        chunk_similarity_top_percent=args.chunk_similarity_top_percent,
        chunk_similarity_max_tokens_per_chunk=args.chunk_similarity_max_tokens_per_chunk,
        chunk_similarity_max_neighbors=args.chunk_similarity_max_neighbors,
        use_crossfile_chunk_similarity=args.use_crossfile_chunk_similarity,
        crossfile_chunk_similarity_top_percent=args.crossfile_chunk_similarity_top_percent,
    )
    print("Model patched successfully\n")

    if args.use_compile:
        print(f"Applying torch.compile (mode={args.compile_mode})...")
        print("Note: the first run will be slower because kernels are compiled on demand.")
        if hasattr(torch, '_dynamo'):
            torch._dynamo.config.suppress_errors = False
        model = torch.compile(model, mode=args.compile_mode)
        print("torch.compile applied successfully\n")

    if not args.no_warmup:
        warmup_model(model, tokenizer, device=device)

    print(f"Loading dataset from {args.data_path}...")
    dataset = load_lcc_dataset(args.data_path, max_samples=args.max_samples)
    print(f"Loaded {len(dataset)} samples")

    dataset_type, language = detect_dataset_type(args.data_path)
    print(f"Detected dataset type: {dataset_type.upper()}")
    if language:
        print(f"Detected language: {language}")
    if dataset_type == 'cceval':
        print("Using CCEval-specific metrics.")
    print()

    precomputed_blocks = None
    skip_precompute = False
    if args.precomputed_blocks_path:
        if os.path.exists(args.precomputed_blocks_path):
            print(f"Loading precomputed blocks from {args.precomputed_blocks_path}")
            precomputed_blocks = torch.load(args.precomputed_blocks_path)
            skip_precompute = True
        else:
            print(f"WARNING: precomputed_blocks_path not found: {args.precomputed_blocks_path}")

    embedding_layer = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    embedding_weight = getattr(embedding_layer, "weight", None) if embedding_layer is not None else None
    if args.use_chunk_similarity and embedding_weight is None:
        raise ValueError("use_chunk_similarity=True requires model.get_input_embeddings().weight to be available.")

    evaluator = SabreCoderEvaluatorTriton(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        use_truncation=True,
        verbose=True,
        block_size=args.block_size,
        window_size=args.window_size,
        num_prefix_tokens=args.num_prefix_tokens,
        num_suffix_tokens=args.num_suffix_tokens,
        global_last_k_chunks=args.global_last_k_chunks,
        max_chunk_tokens=args.max_chunk_tokens,
        precomputed_blocks=precomputed_blocks,
        skip_precompute=skip_precompute,
        dataset_type=dataset_type,
        language=language,
        crossfile_full_attention=args.crossfile_full_attention,
        embedding_weight=embedding_weight,
        use_chunk_similarity=args.use_chunk_similarity,
        chunk_similarity_top_percent=args.chunk_similarity_top_percent,
        chunk_similarity_max_tokens_per_chunk=args.chunk_similarity_max_tokens_per_chunk,
        chunk_similarity_max_neighbors=args.chunk_similarity_max_neighbors,
        use_crossfile_chunk_similarity=args.use_crossfile_chunk_similarity,
        crossfile_chunk_similarity_top_percent=args.crossfile_chunk_similarity_top_percent,
    )

    print("Starting evaluation...\n")
    metrics = evaluator.evaluate(dataset)

    call_counts = get_attention_call_counts()
    print("\nAttention call statistics:")
    print(f"  Sparse calls: {call_counts['sparse']}")
    print(f"  Dense calls: {call_counts['dense']}")

    sparsity_stats = get_sparsity_stats()
    print("\nSparsity statistics:")
    print(f"  Total samples: {sparsity_stats['num_samples']}")
    print(f"  Total causal blocks: {sparsity_stats['total_blocks']:,}")
    print(f"  Active causal blocks: {sparsity_stats['active_blocks']:,}")
    print(f"  Sparsity ratio: {sparsity_stats['sparsity_ratio']*100:.2f}%")
    print(f"  Attend ratio: {sparsity_stats['attend_ratio']*100:.2f}%")

    print_cache_stats()

    metrics['sparsity_stats'] = sparsity_stats

    print("\n" + "="*60)
    print("RESULTS - SabreCoder (Triton)")
    print("="*60)
    print(f"Samples evaluated: {metrics['num_samples']}")
    print(f"\nLatency:")
    print(f"  Avg prefill time: {metrics['avg_prefill_time']*1000:.2f} ms")
    print(f"  Avg generation time: {metrics['avg_generation_time']*1000:.2f} ms")
    print(f"  Avg TTFT: {metrics['avg_ttft_ms']:.2f} ms")
    print(f"  Prefill throughput: {metrics['prefill_throughput']:.2f} tokens/s")
    print(f"  Generation throughput: {metrics['generation_throughput']:.2f} tokens/s")
    print(f"\nQuality:")
    print(f"  Input PPL: {metrics['avg_input_ppl']:.4f}")
    print(f"  Generation PPL: {metrics['avg_generation_ppl']:.4f}")
    print(f"  Exact Match: {metrics['avg_exact_match']:.2f}%")
    print(f"  F1 Score: {metrics['avg_f1']:.2f}%")
    print(f"  Edit Similarity: {metrics['avg_edit_similarity']:.2f}%")
    print(f"\nMemory:")
    print(f"  Avg peak memory: {metrics['avg_memory_mb']:.2f} MB")
    print("="*60 + "\n")

    print("Saving detailed results...")
    dataset_name = os.path.basename(args.data_path).replace('.jsonl', '').replace('_filtered', '').replace('_longprompt', '')
    method_dir = "sabrecoder"
    config = {
        'model_name': args.model_name,
        'max_length': args.max_length,
        'max_new_tokens': args.max_new_tokens,
        'window_size': args.window_size,
        'num_prefix_tokens': args.num_prefix_tokens,
        'num_suffix_tokens': args.num_suffix_tokens,
        'implementation': 'triton_block_sparse',
        'use_sparse_for_generation': args.use_sparse_for_generation,
    }

    extended_sparsity_stats = dict(sparsity_stats) if sparsity_stats else {}
    extended_sparsity_stats['attention_calls'] = call_counts

    output_paths = save_unified_results(
        metrics=metrics,
        method_name="SabreCoder",
        method_dir=method_dir,
        dataset_name=dataset_name,
        config=config,
        output_dir=args.output_dir,
        sparsity_stats=extended_sparsity_stats
    )

    print("\nEvaluation complete.")
    print("Output files:")
    print(f"  Results: {output_paths['benchmark']}")
    print(f"  Details: {output_paths['detailed']}")
    print(f"  Summary: {output_paths['summary']}")
    print()


if __name__ == '__main__':
    main()
