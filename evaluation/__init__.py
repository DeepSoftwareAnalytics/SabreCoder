"""Shared evaluation utilities for SabreCoder benchmarks."""

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

from .utils import (
    load_lcc_dataset,
    detect_dataset_type,
    warmup_model,
    format_results_dict,
    save_results,
    print_results_table,
    get_device_info
)

from .evaluator import UnifiedEvaluator

__all__ = [
    # Metrics
    'compute_exact_match',
    'compute_f1',
    'compute_edit_similarity',
    'compute_perplexity',
    'truncate_to_answer_length',
    'aggregate_metrics',
    'compute_cceval_exact_match',
    'compute_cceval_identifier_f1',
    'compute_cceval_edit_similarity',
    # Utils
    'load_lcc_dataset',
    'detect_dataset_type',
    'warmup_model',
    'format_results_dict',
    'save_results',
    'print_results_table',
    'get_device_info',
    # Evaluator
    'UnifiedEvaluator',
]
