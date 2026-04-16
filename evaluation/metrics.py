"""
Shared evaluation metrics for all sparse attention methods.

Includes:
- Exact Match (EM)
- F1 Score (token-level for LCC, identifier-level for CCEval)
- Edit Similarity (ES)
- Perplexity (PPL)
- Answer truncation utility
"""
import re
import torch
import numpy as np
from collections import Counter


def normalize_answer(s):
    """Normalize answer text for evaluation."""
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_extra_whitespace(text):
        return re.sub(r'\s+', ' ', text).strip()

    return remove_extra_whitespace(white_space_fix(s))


def truncate_to_answer_length(generated_text, answers, tokenizer):
    """
    Truncate generated text to match answer length.

    This is critical for evaluation consistency: we only score the first N tokens
    where N is the length of the ground truth answer.

    Args:
        generated_text: Model-generated text
        answers: List of ground truth answers
        tokenizer: Tokenizer for encoding/decoding

    Returns:
        Truncated generated text
    """
    if not answers or not generated_text:
        return generated_text

    # Use first answer's length as target
    answer_tokens = tokenizer.encode(answers[0], add_special_tokens=False)
    target_length = len(answer_tokens)

    if target_length == 0:
        return generated_text

    # Tokenize generated text and truncate
    generated_tokens = tokenizer.encode(generated_text, add_special_tokens=False)
    truncated_tokens = generated_tokens[:target_length]

    # Decode back to text
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)

    return truncated_text


def compute_exact_match(prediction, ground_truths):
    """
    Compute Exact Match score.

    Args:
        prediction: Predicted text (should be truncated first!)
        ground_truths: List of acceptable answers

    Returns:
        1.0 if exact match found, 0.0 otherwise
    """
    normalized_pred = normalize_answer(prediction)
    for ground_truth in ground_truths:
        normalized_gt = normalize_answer(ground_truth)
        if normalized_pred == normalized_gt:
            return 1.0
    return 0.0


def compute_f1(prediction, ground_truths):
    """
    Compute token-level F1 score.

    Args:
        prediction: Predicted text (should be truncated first!)
        ground_truths: List of acceptable answers

    Returns:
        Maximum F1 score across all ground truths
    """
    def get_tokens(text):
        return normalize_answer(text).split()

    pred_tokens = get_tokens(prediction)
    if not pred_tokens:
        return 0.0

    max_f1 = 0.0
    for ground_truth in ground_truths:
        gt_tokens = get_tokens(ground_truth)
        if not gt_tokens:
            continue

        # Count common tokens
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            f1 = 0.0
        else:
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gt_tokens)
            f1 = (2 * precision * recall) / (precision + recall)

        max_f1 = max(max_f1, f1)

    return max_f1


def compute_edit_distance(s1, s2):
    """Compute Levenshtein edit distance."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]


def compute_edit_similarity(prediction, ground_truths):
    """
    Compute edit similarity (1 - normalized edit distance).

    Args:
        prediction: Predicted text (should be truncated first!)
        ground_truths: List of acceptable answers

    Returns:
        Maximum edit similarity score
    """
    normalized_pred = normalize_answer(prediction)

    max_similarity = 0.0
    for ground_truth in ground_truths:
        normalized_gt = normalize_answer(ground_truth)

        if not normalized_pred and not normalized_gt:
            similarity = 1.0
        elif not normalized_pred or not normalized_gt:
            similarity = 0.0
        else:
            edit_dist = compute_edit_distance(normalized_pred, normalized_gt)
            max_len = max(len(normalized_pred), len(normalized_gt))
            similarity = 1.0 - (edit_dist / max_len)

        max_similarity = max(max_similarity, similarity)

    return max_similarity


def compute_perplexity(model, input_ids, attention_mask=None, device='cuda'):
    """
    Compute perplexity for a given input.

    Args:
        model: Language model
        input_ids: Input token IDs (Tensor)
        attention_mask: Optional attention mask
        device: Device to run on

    Returns:
        Perplexity value (float)
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        loss = outputs.loss
        ppl = torch.exp(loss).item()

        # Filter out invalid values
        if np.isnan(ppl) or np.isinf(ppl):
            return float('inf')

        return ppl


def aggregate_metrics(metric_list):
    """
    Aggregate a list of metric values.

    Args:
        metric_list: List of metric values

    Returns:
        Mean of valid values, or 0.0 if empty
    """
    if not metric_list:
        return 0.0

    valid_values = [v for v in metric_list if not np.isnan(v) and not np.isinf(v)]
    if not valid_values:
        return 0.0

    return np.mean(valid_values)


# ============================================================================
# CCEval-specific metrics (for code completion evaluation)
# ============================================================================

def remove_comments_simple(code):
    """Remove simple comments from code (Python and Java style)."""
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'//.*', '', code)
    return code


def extract_identifiers_simple(source_code, language=None):
    """
    Extract identifiers from source code.
    
    This is a simplified version that doesn't require tree-sitter.
    For full accuracy, use the CCEval evaluation script with tree-sitter.
    
    Args:
        source_code: Source code string
        language: Programming language (optional, for keyword filtering)
    
    Returns:
        List of identifiers
    """
    # Remove strings to avoid extracting identifiers from string literals
    string_pattern = r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\''
    code_without_strings = re.sub(string_pattern, '', source_code)
    
    # Extract identifier-like tokens
    identifier_pattern = r'[_a-zA-Z][_a-zA-Z0-9]*'
    identifiers = re.findall(identifier_pattern, code_without_strings)
    
    # Filter out common keywords if language is specified
    if language:
        keywords = get_language_keywords_simple(language)
        identifiers = [id for id in identifiers if id not in keywords]
    
    return identifiers


def get_language_keywords_simple(language):
    """Get common keywords for a programming language."""
    keywords = {
        'python': {
            'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
            'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
            'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
            'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
            'try', 'while', 'with', 'yield'
        },
        'java': {
            'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch',
            'char', 'class', 'const', 'continue', 'default', 'do', 'double',
            'else', 'enum', 'extends', 'final', 'finally', 'float', 'for',
            'goto', 'if', 'implements', 'import', 'instanceof', 'int', 'interface',
            'long', 'native', 'new', 'package', 'private', 'protected', 'public',
            'return', 'short', 'static', 'strictfp', 'super', 'switch', 'synchronized',
            'this', 'throw', 'throws', 'transient', 'try', 'void', 'volatile', 'while'
        },
        'csharp': {
            'abstract', 'as', 'base', 'bool', 'break', 'byte', 'case', 'catch',
            'char', 'checked', 'class', 'const', 'continue', 'decimal', 'default',
            'delegate', 'do', 'double', 'else', 'enum', 'event', 'explicit',
            'extern', 'false', 'finally', 'fixed', 'float', 'for', 'foreach',
            'goto', 'if', 'implicit', 'in', 'int', 'interface', 'internal',
            'is', 'lock', 'long', 'namespace', 'new', 'null', 'object', 'operator',
            'out', 'override', 'params', 'private', 'protected', 'public',
            'readonly', 'ref', 'return', 'sbyte', 'sealed', 'short', 'sizeof',
            'stackalloc', 'static', 'string', 'struct', 'switch', 'this', 'throw',
            'true', 'try', 'typeof', 'uint', 'ulong', 'unchecked', 'unsafe',
            'ushort', 'using', 'virtual', 'void', 'volatile', 'while'
        }
    }
    return keywords.get(language, set())


def compute_cceval_edit_similarity(prediction, ground_truths):
    """
    Compute edit similarity using fuzzywuzzy ratio (CCEval style).
    
    This uses the same method as CCEval benchmark.
    
    Args:
        prediction: Predicted text
        ground_truths: List of acceptable answers
    
    Returns:
        Maximum edit similarity score (0-100)
    """
    try:
        from fuzzywuzzy import fuzz
        
        pred = prediction.strip()
        max_ratio = 0.0
        
        for gt in ground_truths:
            gt = gt.strip()
            ratio = fuzz.ratio(pred, gt)
            max_ratio = max(max_ratio, ratio)
        
        return max_ratio
    except ImportError:
        # Fallback to Levenshtein-based method if fuzzywuzzy not available
        return compute_edit_similarity(prediction, ground_truths) * 100


def compute_cceval_exact_match(prediction, ground_truths, language=None):
    """
    Compute exact match for code (CCEval style).
    
    Compares code line-by-line after removing comments.
    
    Args:
        prediction: Predicted code
        ground_truths: List of acceptable code answers
        language: Programming language (optional)
    
    Returns:
        1.0 if exact match found, 0.0 otherwise
    """
    pred = remove_comments_simple(prediction)
    pred_lines = [l.strip() for l in pred.split("\n") if l.strip()]
    
    for ground_truth in ground_truths:
        gt = remove_comments_simple(ground_truth)
        gt_lines = [l.strip() for l in gt.split("\n") if l.strip()]
        
        if pred_lines == gt_lines:
            return 1.0
    
    return 0.0


def compute_cceval_identifier_f1(prediction, ground_truths, language=None):
    """
    Compute identifier-level F1 score (CCEval style).
    
    This computes F1 based on identifiers (variable names, function names, etc.)
    rather than all tokens.
    
    Args:
        prediction: Predicted code
        ground_truths: List of acceptable code answers
        language: Programming language for keyword filtering
    
    Returns:
        Maximum identifier F1 score (0-1)
    """
    pred_ids = extract_identifiers_simple(prediction, language)
    
    if not pred_ids:
        return 0.0
    
    max_f1 = 0.0
    
    for ground_truth in ground_truths:
        gt_ids = extract_identifiers_simple(ground_truth, language)
        
        if not gt_ids:
            continue
        
        # Compute identifier match
        pred_id_set = set(pred_ids)
        gt_id_set = set(gt_ids)
        
        tp = len(pred_id_set & gt_id_set)
        fp = len(pred_id_set - gt_id_set)
        fn = len(gt_id_set - pred_id_set)
        
        if tp == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        max_f1 = max(max_f1, f1)
    
    return max_f1
