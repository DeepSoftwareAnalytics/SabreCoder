#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${MODEL:-deepseek-ai/deepseek-coder-1.3b-base}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/_lcc_budget_prompts/LCC_python_test_ctx_12288_14336.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
USE_TOKEN_LEVEL_SPARSITY="${USE_TOKEN_LEVEL_SPARSITY:-false}"

echo "=========================================="
echo "SabreCoder evaluation"
echo "=========================================="
echo "Model:        $MODEL"
echo "Dataset:      $DATA_PATH"
echo "Max samples:  $MAX_SAMPLES"
echo "Max length:   $MAX_LENGTH"
echo "Output dir:   $OUTPUT_DIR"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
fi
if [[ "$USE_TOKEN_LEVEL_SPARSITY" == "true" ]]; then
    echo "Sparsity:     token-level"
else
    echo "Sparsity:     block-level"
fi
echo "=========================================="
echo

CMD=(
    python "${SCRIPT_DIR}/run_eval.py"
    --model_name "$MODEL"
    --data_path "$DATA_PATH"
    --output_dir "$OUTPUT_DIR"
    --max_samples "$MAX_SAMPLES"
    --max_length "$MAX_LENGTH"
    --max_new_tokens "$MAX_NEW_TOKENS"
)

if [[ "$USE_TOKEN_LEVEL_SPARSITY" == "true" ]]; then
    CMD+=(--use_token_level_sparsity)
fi

"${CMD[@]}"

echo
echo "Done."
echo "Results:"
echo "  $OUTPUT_DIR/sabrecoder/<dataset>/results.json"
echo "  $OUTPUT_DIR/sabrecoder/<dataset>/detailed.json"
echo "  $OUTPUT_DIR/sabrecoder/<dataset>/summary.txt"
