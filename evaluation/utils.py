"""
Shared utilities for evaluation.

Includes:
- Dataset loading
- Warmup utilities
- Result formatting
"""
import json
import torch
import time


def detect_dataset_type(file_path):
    """
    Detect dataset type from file path.
    
    Args:
        file_path: Path to dataset file
    
    Returns:
        Tuple of (dataset_type, language)
        - dataset_type: 'cceval' or 'lcc'
        - language: 'python', 'java', 'csharp', or None
    """
    file_path_lower = file_path.lower()
    
    # Detect dataset type
    if 'cceval' in file_path_lower:
        dataset_type = 'cceval'
    elif 'lcc' in file_path_lower:
        dataset_type = 'lcc'
    else:
        dataset_type = 'lcc'
    
    # Detect language
    language = None
    if 'python' in file_path_lower:
        language = 'python'
    elif 'java' in file_path_lower:
        language = 'java'
    elif 'csharp' in file_path_lower or 'c#' in file_path_lower:
        language = 'csharp'
    
    return dataset_type, language


def load_lcc_dataset(file_path, max_samples=None):
    """
    Load LCC dataset from JSONL file.

    Args:
        file_path: Path to .jsonl file
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        List of dataset items
    """
    def _normalize_item(item: dict) -> dict:
        # Normalize prompt field
        if 'context' not in item:
            if 'prompt' in item:
                item['context'] = item.get('prompt', '')
            elif 'input' in item:
                item['context'] = item.get('input', '')

        # Normalize answer field for UnifiedEvaluator (expects 'answers')
        if 'answers' not in item:
            if 'ground_truth' in item:
                gt = item.get('ground_truth', '')
                item['answers'] = gt if isinstance(gt, list) else [gt]
            elif 'groundtruth' in item:
                gt = item.get('groundtruth', '')
                item['answers'] = gt if isinstance(gt, list) else [gt]
            elif 'gt' in item:
                gt = item.get('gt', '')
                item['answers'] = gt if isinstance(gt, list) else [gt]

        return item

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            data.append(_normalize_item(item))
    return data


def warmup_model(model, tokenizer, device='cuda', num_runs=3, num_tokens=16):
    """
    Warmup model to ensure fair timing measurements.

    Args:
        model: Language model
        tokenizer: Tokenizer
        device: Device to run on
        num_runs: Number of warmup runs
        num_tokens: Number of tokens to generate per run
    """
    print(f"\nPerforming warmup ({num_runs} runs, {num_tokens} tokens)...")

    warmup_text = "def hello_world():"
    warmup_inputs = tokenizer(warmup_text, return_tensors='pt').to(device)

    for i in range(num_runs):
        with torch.no_grad():
            _ = model.generate(
                **warmup_inputs,
                max_new_tokens=num_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        if device == 'cuda':
            torch.cuda.synchronize()

    print("Warmup complete\n")


def format_results_dict(metrics, method_name):
    """
    Format metrics into a standardized results dictionary.

    Args:
        metrics: Dictionary of raw metrics
        method_name: Name of the method

    Returns:
        Formatted results dictionary
    """
    return {
        'method': method_name,
        'num_samples': metrics.get('num_samples', 0),
        'latency': {
            'avg_prefill_ms': metrics.get('avg_prefill_time', 0) * 1000,
            'avg_generation_ms': metrics.get('avg_generation_time', 0) * 1000,
            'avg_ttft_ms': metrics.get('avg_ttft_ms', 0),
            'prefill_throughput': metrics.get('prefill_throughput', 0),
            'generation_throughput': metrics.get('generation_throughput', 0),
        },
        'quality': {
            'avg_input_ppl': metrics.get('avg_input_ppl', float('inf')),
            'avg_generation_ppl': metrics.get('avg_generation_ppl', float('inf')),
            'exact_match': metrics.get('avg_exact_match', 0),
            'f1_score': metrics.get('avg_f1', 0),
            'edit_similarity': metrics.get('avg_edit_similarity', 0),
        },
        'sparsity': {
            'sparsity_ratio': metrics.get('sparsity', 0),
            'attend_ratio': metrics.get('attend_ratio', 1.0),
        },
        'memory': {
            'avg_memory_mb': metrics.get('avg_memory_mb', 0),
        }
    }


def save_results(results, output_path):
    """
    Save results to JSON file.

    Args:
        results: Results dictionary
        output_path: Path to save to
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")


def print_results_table(results_list):
    """
    Print a results table of evaluated runs.

    Args:
        results_list: List of result dictionaries from evaluated runs
    """
    if not results_list:
        print("No results to display")
        return

    # Build header first so we can derive a stable table width.
    print("BENCHMARK RESULTS")

    # Header (removed PPL(in) column)
    header = (
        f"{'Method':<25} | {'Prefill(ms)':<12} | {'Gen(ms)':<10} | {'TTFT(ms)':<10} | "
        f"{'PPL(gen)':<10} | {'EM(%)':<7} | {'F1(%)':<7} | {'ES(%)':<7} | {'Sparsity(%)':<12} | "
        f"{'AvgMem(MB)':<10} | {'Samples':<7}"
    )
    table_width = len(header)

    print("\n" + "=" * table_width)
    print("BENCHMARK RESULTS")
    print("=" * table_width)
    print(header)
    print("-" * table_width)

    # Rows
    for result in results_list:
        method = result['method']
        latency = result['latency']
        quality = result['quality']
        sparsity = result.get('sparsity', {})
        memory = result.get('memory', {})

        # Get sparsity ratio with fallback
        sparsity_ratio = sparsity.get('sparsity_ratio', 0.0) if sparsity else 0.0
        avg_mem_mb = float(memory.get('avg_memory_mb', 0.0) or 0.0)
        n_samples = int(result.get('num_samples', 0) or 0)

        # Get generation PPL with fallback
        gen_ppl = quality.get('avg_generation_ppl', float('inf'))
        # Format generation PPL: if too large, show in scientific notation
        if gen_ppl > 9999:
            gen_ppl_str = f"{gen_ppl:>10.2e}"
        else:
            gen_ppl_str = f"{gen_ppl:>10.2f}"

        row = f"{method:<25} | " \
              f"{latency['avg_prefill_ms']:>11.2f} | " \
              f"{latency['avg_generation_ms']:>9.2f} | " \
              f"{latency['avg_ttft_ms']:>9.2f} | " \
              f"{gen_ppl_str} | " \
              f"{quality['exact_match']:>6.2f} | " \
              f"{quality['f1_score']:>6.2f} | " \
              f"{quality['edit_similarity']:>6.2f} | " \
              f"{sparsity_ratio*100:>11.2f} | " \
              f"{avg_mem_mb:>10.2f} | " \
              f"{n_samples:>7d}"

        print(row)

    print("=" * table_width)

    # Summary statistics
    print("\nSummary:")
    samples_list = [int(r.get('num_samples', 0) or 0) for r in results_list]
    if samples_list:
        print(f"  Samples evaluated (min/max): {min(samples_list)}/{max(samples_list)}")

    # Find best runs
    best_prefill = min(results_list, key=lambda x: x['latency']['avg_prefill_ms'])
    best_quality = max(results_list, key=lambda x: x['quality']['exact_match'])

    # Find best sparsity (only among methods that have sparsity info)
    methods_with_sparsity = [r for r in results_list if r.get('sparsity') and r['sparsity'].get('sparsity_ratio', 0) > 0]
    if methods_with_sparsity:
        best_sparsity = max(methods_with_sparsity, key=lambda x: x['sparsity']['sparsity_ratio'])

    print(f"  Fastest prefill: {best_prefill['method']} ({best_prefill['latency']['avg_prefill_ms']:.2f}ms)")
    print(f"  Best quality (EM): {best_quality['method']} ({best_quality['quality']['exact_match']:.2f}%)")
    if methods_with_sparsity:
        print(f"  Highest sparsity: {best_sparsity['method']} ({best_sparsity['sparsity']['sparsity_ratio']*100:.2f}%)")

    print("\n")


def get_device_info():
    """Get device information for benchmarking."""
    if torch.cuda.is_available():
        device = 'cuda'
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Device: {device_name}")
        print(f"Total GPU Memory: {total_memory:.2f} GB")
    else:
        device = 'cpu'
        print("Device: CPU")

    return device
