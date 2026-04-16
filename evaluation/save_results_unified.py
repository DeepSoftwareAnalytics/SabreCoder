"""Utilities for writing benchmark outputs in a consistent layout."""
import os


def save_unified_results(
    metrics,
    method_name,
    method_dir,
    dataset_name,
    config,
    output_dir,
    sparsity_stats=None,
    mask_info=None,
    viz_paths=None
):
    """Save benchmark, detailed, and summary outputs for a single dataset run."""
    base_dir = os.path.join(output_dir, method_dir, dataset_name)
    os.makedirs(base_dir, exist_ok=True)

    from evaluation.utils import format_results_dict, save_results

    results = format_results_dict(metrics, method_name)
    if sparsity_stats is not None:
        sparsity_value = sparsity_stats.get('sparsity_ratio', sparsity_stats.get('sparsity', 0.0))
        results['sparsity'] = {
            'sparsity_ratio': sparsity_value,
            'attend_ratio': sparsity_stats.get('attend_ratio', 1.0)
        }
    else:
        results['sparsity'] = {
            'sparsity_ratio': 0.0,
            'attend_ratio': 1.0
        }

    benchmark_path = os.path.join(base_dir, 'results.json')
    save_results(results, benchmark_path)

    detailed_results = {
        'method': method_name,
        'configuration': config,
        'overall_stats': {
            'num_samples': metrics['num_samples'],
            'avg_prefill_ms': metrics['avg_prefill_time'] * 1000,
            'avg_generation_ms': metrics['avg_generation_time'] * 1000,
            'avg_ttft_ms': metrics['avg_ttft_ms'],
            'avg_input_ppl': metrics['avg_input_ppl'],
            'avg_generation_ppl': metrics['avg_generation_ppl'],
            'avg_exact_match': metrics['avg_exact_match'],
            'avg_f1': metrics['avg_f1'],
            'avg_edit_similarity': metrics['avg_edit_similarity'],
        },
        'per_sample_details': metrics['sample_details'],
    }

    if sparsity_stats is not None:
        detailed_results['sparsity_stats'] = sparsity_stats

    if mask_info is not None:
        detailed_results['attention_mask_info'] = mask_info
        if viz_paths:
            detailed_results['attention_mask_info']['visualization_files'] = {
                k: os.path.basename(v) for k, v in viz_paths.items()
            }

    detailed_path = os.path.join(base_dir, 'detailed.json')
    save_results(detailed_results, detailed_path)

    summary_path = os.path.join(base_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"{method_name.upper()} - EVALUATION SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write(f"Configuration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        f.write(f"Overall Results ({metrics['num_samples']} samples):\n")
        f.write(f"  Latency:\n")
        f.write(f"    Avg prefill: {metrics['avg_prefill_time']*1000:.2f} ms\n")
        f.write(f"    Avg generation: {metrics['avg_generation_time']*1000:.2f} ms\n")
        f.write(f"    Avg TTFT: {metrics['avg_ttft_ms']:.2f} ms\n")
        f.write(f"  Quality:\n")
        f.write(f"    Input PPL: {metrics['avg_input_ppl']:.4f}\n")
        f.write(f"    Generation PPL: {metrics['avg_generation_ppl']:.4f}\n")
        f.write(f"    Exact Match: {metrics['avg_exact_match']:.2f}%\n")
        f.write(f"    F1 Score: {metrics['avg_f1']:.2f}%\n")
        f.write(f"    Edit Similarity: {metrics['avg_edit_similarity']:.2f}%\n")

        if sparsity_stats is not None:
            sparsity_value = sparsity_stats.get('sparsity_ratio', sparsity_stats.get('sparsity', 0.0))
            f.write(f"  Sparsity:\n")
            f.write(f"    Sparsity ratio: {sparsity_value*100:.2f}%\n")
            f.write(f"    Attend ratio: {sparsity_stats.get('attend_ratio', 1.0)*100:.2f}%\n")

        f.write("\n")
        f.write("="*80 + "\n")
        f.write("PER-SAMPLE DETAILS\n")
        f.write("="*80 + "\n\n")

        for sample in metrics['sample_details']:
            f.write(f"Sample {sample['sample_idx']}:\n")
            f.write(f"  Input length: {sample['input_length']} tokens\n")
            f.write(f"  Output length: {sample['output_length']} tokens\n")
            f.write(f"  Prefill time: {sample['prefill_time']*1000:.2f} ms\n")
            f.write(f"  Generation time: {sample['generation_time']*1000:.2f} ms\n")
            f.write(f"  TTFT: {sample['ttft_ms']:.2f} ms\n")
            f.write(f"  Input PPL: {sample['input_ppl']:.4f}\n")
            f.write(f"  Generation PPL: {sample['generation_ppl']:.4f}\n")
            f.write(f"  EM: {sample['em']:.2f}\n")
            f.write(f"  F1: {sample['f1']:.2f}\n")
            f.write(f"  ES: {sample['es']:.2f}\n")
            f.write(f"  Generated (first 200 chars): {sample['generated_text'][:200]}...\n")
            f.write(f"  Truncated (first 200 chars): {sample['truncated_text'][:200]}...\n")
            f.write(f"  Ground truth: {sample['ground_truth']}\n")
            f.write("-"*80 + "\n\n")

    return {
        'benchmark': benchmark_path,
        'detailed': detailed_path,
        'summary': summary_path,
        'visualizations': viz_paths or {}
    }
