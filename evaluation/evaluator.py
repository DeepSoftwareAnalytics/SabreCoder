"""
Shared evaluator for SabreCoder experiments.
"""
import torch
import time
import numpy as np
from tqdm import tqdm

from .metrics import (
    compute_exact_match,
    compute_f1,
    compute_edit_similarity,
    compute_perplexity,
    truncate_to_answer_length,
    aggregate_metrics,
    compute_cceval_exact_match,
    compute_cceval_identifier_f1,
    compute_cceval_edit_similarity
)


class UnifiedEvaluator:
    """
    Unified evaluator for SabreCoder benchmark runs.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device='cuda',
        max_length=4096,
        max_new_tokens=64,
        use_truncation=True,
        verbose=True,
        dataset_type='lcc',
        language=None,
        crossfile_module=None
    ):
        """
        Initialize evaluator.

        Args:
            model: Language model (can be patched with sparse attention)
            tokenizer: Tokenizer
            device: Device to run on
            max_length: Maximum input length
            max_new_tokens: Maximum tokens to generate
            use_truncation: Whether to truncate generated text to answer length
            verbose: Whether to print progress
            dataset_type: Type of dataset ('lcc' or 'cceval')
            language: Programming language for CCEval datasets (python/java/csharp)
            crossfile_module: Optional module with set_crossfile_len() for V1 methods
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.use_truncation = use_truncation
        self.verbose = verbose
        self.dataset_type = dataset_type.lower()
        self.language = language
        self.crossfile_module = crossfile_module

        self.model.eval()

    def evaluate(self, dataset, method_specific_setup=None):
        """
        Evaluate model on dataset.

        Args:
            dataset: List of dataset items (each with 'context' and 'answers')
            method_specific_setup: Optional callable to run before each sample
                                  (e.g., for precomputing embeddings in learned method)

        Returns:
            Dictionary of metrics
        """
        # Statistics
        total_prefill_time = 0
        total_generation_time = 0
        total_ttft = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_memory = 0
        num_samples = 0
        skipped_samples = 0

        # Quality metrics
        input_ppls = []
        generation_ppls = []
        exact_match_scores = []
        f1_scores = []
        edit_similarity_scores = []

        # Sample details
        sample_details = []

        if self.verbose:
            print(f"\nEvaluating {len(dataset)} samples...")

        with torch.no_grad():
            for sample_idx, item in enumerate(tqdm(dataset, desc="Evaluation", disable=not self.verbose)):
                context = item.get('context', '')
                answers = item.get('answers', [])

                # Ensure answers is always a list (CCEval datasets have string answers)
                if isinstance(answers, str):
                    answers = [answers]

                # Skip empty samples
                if not context:
                    skipped_samples += 1
                    continue

                # 🔥 V1 Crossfile Support: Set crossfile length if available
                crossfile_len = 0
                if self.crossfile_module is not None and 'crossfile_context' in item:
                    crossfile_context = item.get('crossfile_context', '')
                    if crossfile_context:
                        # Tokenize crossfile to get its length
                        crossfile_tokens = self.tokenizer(
                            crossfile_context,
                            return_tensors='pt',
                            add_special_tokens=False
                        )
                        crossfile_len = crossfile_tokens['input_ids'].shape[1]

                        # Set crossfile length in the module
                        if hasattr(self.crossfile_module, 'set_crossfile_len'):
                            self.crossfile_module.set_crossfile_len(crossfile_len)

                # Tokenize input
                inputs = self.tokenizer(
                    context,
                    return_tensors='pt',
                    max_length=self.max_length,
                    truncation=True,
                    padding=False
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                input_length = inputs['input_ids'].shape[1]

                # Reset memory stats
                if self.device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

                try:
                    # Method-specific setup (e.g., precompute embeddings)
                    if method_specific_setup is not None:
                        method_specific_setup(inputs['input_ids'])

                    # === PREFILL PHASE ===
                    ttft_start = time.time()
                    prefill_start = time.time()

                    outputs = self.model(
                        **inputs,
                        use_cache=True,
                        return_dict=True
                    )

                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    prefill_time = time.time() - prefill_start

                    # Get first token logits and past_key_values for generation
                    past_key_values = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    ttft = time.time() - ttft_start

                    # === GENERATION PHASE ===
                    # Manual generation loop to reuse KV cache from prefill
                    generation_start = time.time()

                    generated_tokens = [next_token]
                    current_token = next_token

                    # Initialize attention_mask for generation
                    if 'attention_mask' in inputs:
                        attention_mask = inputs['attention_mask']
                    else:
                        attention_mask = torch.ones_like(inputs['input_ids'])

                    for step in range(self.max_new_tokens - 1):
                        # Check for EOS
                        if current_token.item() == self.tokenizer.eos_token_id:
                            break

                        # Extend attention mask for new token
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
                        ], dim=1)

                        # Generate next token using cached KV
                        step_outputs = self.model(
                            input_ids=current_token,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            use_cache=True,
                            return_dict=True
                        )

                        past_key_values = step_outputs.past_key_values
                        next_logits = step_outputs.logits[:, -1, :]
                        current_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                        generated_tokens.append(current_token)

                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    generation_time = time.time() - generation_start

                    # Combine generated tokens
                    generated_ids = torch.cat(generated_tokens, dim=1)
                    num_generated = generated_ids.shape[1]

                    # Decode text
                    generated_text = self.tokenizer.decode(
                        generated_ids[0],
                        skip_special_tokens=True
                    )

                    # Compute generation PPL (only on generated tokens)
                    # Use generated tokens (measures model's confidence in its own generation)
                    full_sequence = torch.cat([inputs['input_ids'], generated_ids], dim=1)

                    # Create labels that only compute loss on generated tokens
                    labels = full_sequence.clone()
                    labels[:, :input_length] = -100  # Ignore input tokens in loss calculation

                    gen_outputs = self.model(input_ids=full_sequence, labels=labels)
                    gen_ppl = torch.exp(gen_outputs.loss).item()
                    if not np.isnan(gen_ppl) and not np.isinf(gen_ppl):
                        generation_ppls.append(gen_ppl)

                    # === QUALITY METRICS ===
                    # Truncate before computing metrics.
                    if self.use_truncation:
                        if self.dataset_type == 'cceval':
                            # 🔥 CCEval: Use statement-based truncation (stop at first newline/statement)
                            from .cceval_utils import postprocess_code_lines
                            truncated_text = postprocess_code_lines(
                                context,  # prompt
                                generated_text,  # completion
                                self.language or 'python',  # language
                                use_ast=False  # Use simple newline truncation for speed
                            )
                        else:
                            # LCC: Use token-length based truncation
                            truncated_text = truncate_to_answer_length(
                                generated_text,
                                answers,
                                self.tokenizer
                            )
                    else:
                        truncated_text = generated_text

                    # Compute metrics based on dataset type
                    if self.dataset_type == 'cceval':
                        # CCEval uses code-specific metrics
                        em_score = compute_cceval_exact_match(truncated_text, answers, self.language)
                        f1_score = compute_cceval_identifier_f1(truncated_text, answers, self.language)
                        es_score = compute_cceval_edit_similarity(truncated_text, answers) / 100.0  # Normalize to 0-1
                    else:
                        # LCC uses token-level metrics
                        em_score = compute_exact_match(truncated_text, answers)
                        f1_score = compute_f1(truncated_text, answers)
                        es_score = compute_edit_similarity(truncated_text, answers)

                    exact_match_scores.append(em_score)
                    f1_scores.append(f1_score)
                    edit_similarity_scores.append(es_score)

                    # Update statistics
                    total_prefill_time += prefill_time
                    total_generation_time += generation_time
                    total_ttft += ttft
                    total_input_tokens += input_length
                    total_output_tokens += num_generated
                    num_samples += 1

                    # Memory
                    if self.device == 'cuda':
                        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
                        total_memory += peak_memory

                    # Save sample details
                    sample_details.append({
                        'sample_idx': sample_idx,
                        'input_length': input_length,
                        'output_length': num_generated,
                        'prefill_time': prefill_time,
                        'generation_time': generation_time,
                        'ttft_ms': ttft * 1000,
                        'input_ppl': 0.0,  # No longer computed
                        'generation_ppl': gen_ppl,
                        'generated_text': generated_text,
                        'truncated_text': truncated_text,
                        'ground_truth': answers,
                        'em': em_score,
                        'f1': f1_score,
                        'es': es_score,
                    })

                    if self.verbose and num_samples % 5 == 0:
                        print(f"  Sample {num_samples}: "
                              f"Prefill={prefill_time:.3f}s, "
                              f"Gen={generation_time:.3f}s, "
                              f"EM={em_score:.2f}")

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM on sample {sample_idx}, skipping...")
                        skipped_samples += 1
                        if self.device == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                finally:
                    # 🔥 V1 Crossfile Support: Clear crossfile length after each sample
                    if self.crossfile_module is not None and hasattr(self.crossfile_module, 'clear_crossfile_len'):
                        self.crossfile_module.clear_crossfile_len()

        # Aggregate metrics
        avg_prefill_time = total_prefill_time / num_samples if num_samples > 0 else 0
        avg_generation_time = total_generation_time / num_samples if num_samples > 0 else 0
        avg_ttft = total_ttft / num_samples if num_samples > 0 else 0

        prefill_throughput = total_input_tokens / total_prefill_time if total_prefill_time > 0 else 0
        generation_throughput = total_output_tokens / total_generation_time if total_generation_time > 0 else 0

        avg_memory = total_memory / num_samples if num_samples > 0 else 0
        avg_input_ppl = 0.0  # No longer computed
        avg_gen_ppl = aggregate_metrics(generation_ppls)

        avg_em = np.mean(exact_match_scores) * 100 if exact_match_scores else 0.0
        avg_f1 = np.mean(f1_scores) * 100 if f1_scores else 0.0
        avg_es = np.mean(edit_similarity_scores) * 100 if edit_similarity_scores else 0.0

        return {
            'num_samples': num_samples,
            'skipped_samples': skipped_samples,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_prefill_time': total_prefill_time,
            'avg_prefill_time': avg_prefill_time,
            'prefill_throughput': prefill_throughput,
            'total_generation_time': total_generation_time,
            'avg_generation_time': avg_generation_time,
            'avg_ttft_ms': avg_ttft * 1000,
            'generation_throughput': generation_throughput,
            'avg_memory_mb': avg_memory,
            'avg_input_ppl': avg_input_ppl,
            'avg_generation_ppl': avg_gen_ppl,
            'avg_exact_match': avg_em,
            'avg_f1': avg_f1,
            'avg_edit_similarity': avg_es,
            'sample_details': sample_details,
        }
