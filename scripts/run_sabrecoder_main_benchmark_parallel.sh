#!/usr/bin/env bash
set -euo pipefail
# Run SabreCoder benchmarks across multiple datasets in parallel.
#
# Usage:
#   ./run_sabrecoder_main_benchmark_parallel.sh [options]
#
# Options:
#   --max_samples <N>           Max samples per dataset (default: 400)
#   --cceval_prompt_dir <path>  CCEval prompt dir (default: ../data/_cceval_rag_prompts)
#   --lcc_prompt_dir <path>     LCC prompt dir (default: ../data/_lcc_budget_prompts)
#   --budgets "..."             Budgets list (default: "14336 12288 10240 8192 6144 4096 2048")
#   --datasets <list...>        Explicit dataset keys to run (overrides auto-discovery)
#   --gpus <list>               GPU IDs to use
#   --no_warmup                 Skip warmup runs
#   -h, --help                  Show help

# Default values
MAX_SAMPLES=400
NO_WARMUP=""
MODEL="deepseek-ai/deepseek-coder-1.3b-base"
PROMPT_TOKENIZER=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/results"
LOG_DIR="${ROOT_DIR}/logs"
OUTPUT_BASE_DIR="${ROOT_DIR}/results"
LOG_BASE_DIR="${ROOT_DIR}/logs"
RUN_TAG=""

# If you built prompts with multiple tokenizers (data/*/run.sh --tokenizers ...),
# prompt dirs are suffixed as:
#   data/_cceval_rag_prompts__<tokenizer_slug>
#   data/_lcc_budget_prompts__<tokenizer_slug>
CCEVAL_PROMPT_DIR="${ROOT_DIR}/data/_cceval_rag_prompts"
LCC_PROMPT_DIR="${ROOT_DIR}/data/_lcc_budget_prompts"

BUDGETS="14336 12288 10240 8192 6144 4096 2048"
GPUS=""
DATASET_OVERRIDE=""
CCEVAL_PROMPT_DIR_USER=0
LCC_PROMPT_DIR_USER=0

_detect_default_gpus() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_list
        gpu_list="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | xargs)"
        if [[ -n "$gpu_list" ]]; then
            echo "$gpu_list"
            return 0
        fi
    fi
    echo "0"
}

_slugify_hf_id() {
    local s="$1"
    s="$(echo "$s" | tr '[:upper:]' '[:lower:]')"
    s="${s//\//--}"
    s="$(echo "$s" | sed -E 's/[^a-z0-9]+/-/g; s/-+/-/g; s/^-+//; s/-+$//')"
    if [[ -z "$s" ]]; then
        s="tokenizer"
    fi
    echo "$s"
}

_infer_prompt_tokenizer() {
    local model_lc
    model_lc="$(echo "$1" | tr '[:upper:]' '[:lower:]')"

    if [[ "$model_lc" == *"qwen"* ]]; then
        echo "Qwen/Qwen2.5-Coder-3B"
        return 0
    fi
    if [[ "$model_lc" == *"starcoder2"* ]]; then
        echo "bigcode/starcoder2-3b"
        return 0
    fi
    if [[ "$model_lc" == *"codellama"* ]]; then
        echo "codellama/CodeLlama-7b-hf"
        return 0
    fi

    echo "$1"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            MODEL="$2"
            shift 2
            ;;
        --prompt_tokenizer)
            PROMPT_TOKENIZER="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --cceval_prompt_dir)
            CCEVAL_PROMPT_DIR="$2"
            CCEVAL_PROMPT_DIR_USER=1
            shift 2
            ;;
        --lcc_prompt_dir)
            LCC_PROMPT_DIR="$2"
            LCC_PROMPT_DIR_USER=1
            shift 2
            ;;
        --budgets)
            BUDGETS="$2"
            shift 2
            ;;
        --datasets)
            shift
            # Collect dataset keys until next option (starting with '-') or end of args.
            while [[ $# -gt 0 && "$1" != -* ]]; do
                DATASET_OVERRIDE="${DATASET_OVERRIDE} $1"
                shift
            done
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --output_base_dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --log_base_dir)
            LOG_BASE_DIR="$2"
            shift 2
            ;;
        --run_tag)
            RUN_TAG="$2"
            shift 2
            ;;
        --no_warmup)
            NO_WARMUP="--no_warmup"
            shift
            ;;
        -h|--help)
            echo "Parallel Multi-Dataset Benchmark Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model_name <hf_or_path>   Model name/path (default: $MODEL)"
            echo "  --prompt_tokenizer <hf_id>  Tokenizer used for prompt preprocessing (default: inferred from model)"
            echo "  --max_samples <N>           Max samples per dataset (default: 400)"
            echo "  --cceval_prompt_dir <path>  CCEval prompt dir (default: ../data/_cceval_rag_prompts)"
            echo "  --lcc_prompt_dir <path>     LCC prompt dir (default: ../data/_lcc_budget_prompts)"
            echo "  --budgets \"...\"            Budgets list (default: \"14336 12288 10240 8192 6144 4096 2048\")"
            echo "  --datasets <list...>        Explicit dataset keys to run (overrides auto-discovery)"
            echo "  --gpus <list>               Space-separated GPU IDs (default: auto-detect, fallback \"0\")"
            echo "  --output_base_dir <path>    Base results dir (default: <repo>/results)"
            echo "  --log_base_dir <path>       Base logs dir (default: <repo>/logs)"
            echo "  --run_tag <name>            Subdir name under base dirs (default: <model>__tok_<suffix>_<timestamp>)"
            echo "  --no_warmup                 Skip warmup runs"
            echo "  -h, --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Run SabreCoder on all discovered datasets"
            echo "  $0"
            echo ""
            echo "  # Use only 3 GPUs"
            echo "  $0 --gpus \"0 1 2\""
            echo ""
            echo "  # Custom settings"
            echo "  $0 --max_samples 1000"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [[ -z "$GPUS" ]]; then
    GPUS="$(_detect_default_gpus)"
fi

# If prompt dirs were not explicitly overridden, try to auto-select tokenizer-suffixed dirs.
if [[ -z "$PROMPT_TOKENIZER" ]]; then
    PROMPT_TOKENIZER="$(_infer_prompt_tokenizer "$MODEL")"
fi
TOKENIZER_SUFFIX="$(_slugify_hf_id "$PROMPT_TOKENIZER")"
if [[ "$CCEVAL_PROMPT_DIR_USER" -eq 0 ]]; then
    CCEVAL_CAND="${ROOT_DIR}/data/_cceval_rag_prompts__${TOKENIZER_SUFFIX}"
    if [[ -d "$CCEVAL_CAND" ]]; then
        CCEVAL_PROMPT_DIR="$CCEVAL_CAND"
    fi
fi
if [[ "$LCC_PROMPT_DIR_USER" -eq 0 ]]; then
    LCC_CAND="${ROOT_DIR}/data/_lcc_budget_prompts__${TOKENIZER_SUFFIX}"
    if [[ -d "$LCC_CAND" ]]; then
        LCC_PROMPT_DIR="$LCC_CAND"
    fi
fi

# Place outputs under a dedicated subdirectory (avoid writing directly into <repo>/logs or <repo>/results).
timestamp="$(date +%Y%m%d_%H%M%S)"
MODEL_TAG="${MODEL//\//_}"
MODEL_TAG="${MODEL_TAG// /_}"
if [[ -z "$RUN_TAG" ]]; then
    RUN_TAG="${MODEL_TAG}__tok_${TOKENIZER_SUFFIX}_${timestamp}"
fi
OUTPUT_DIR="${OUTPUT_BASE_DIR}/multi_dataset_parallel/${RUN_TAG}"
LOG_DIR="${LOG_BASE_DIR}/multi_dataset_parallel/${RUN_TAG}"

# Define datasets (CCEval python/java RAG budgets + LCC python/java/csharp buckets)
# Note: `benchmark_multi_dataset.py --datasets lcc` would run LCC sequentially on a single GPU.
# Here we expand group keywords (all/cceval/lcc) into concrete dataset keys so we can run them in parallel.
declare -A _SEEN_DS=()
DATASETS=()

_add_dataset() {
    local ds="$1"
    if [[ -z "${_SEEN_DS[$ds]+x}" ]]; then
        DATASETS+=("$ds")
        _SEEN_DS["$ds"]=1
    fi
}

_discover_cceval() {
    local B
    for B in $BUDGETS; do
        if [[ -f "${CCEVAL_PROMPT_DIR}/cceval_python_rag_${B}.jsonl" ]]; then
            _add_dataset "cceval_python_rag_${B}"
        fi
        if [[ -f "${CCEVAL_PROMPT_DIR}/cceval_java_rag_${B}.jsonl" ]]; then
            _add_dataset "cceval_java_rag_${B}"
        fi
    done
}

_discover_lcc() {
    local B LANG
    for B in $BUDGETS; do
        shopt -s nullglob
        for LANG in python java csharp; do
            local BEST_FILE=""
            local BEST_LOW=-1
            local f BASE TMP LOW HIGH
            for f in "${LCC_PROMPT_DIR}/LCC_${LANG}_test_ctx_"*"_${B}.jsonl"; do
                [[ -f "$f" ]] || continue
                BASE="$(basename "$f")"
                TMP="${BASE#LCC_${LANG}_test_ctx_}"
                TMP="${TMP%.jsonl}"
                LOW="${TMP%_*}"
                HIGH="${TMP##*_}"
                [[ "$HIGH" == "$B" ]] || continue
                if [[ "$LOW" =~ ^[0-9]+$ ]] && (( LOW > BEST_LOW )); then
                    BEST_LOW="$LOW"
                    BEST_FILE="$f"
                fi
            done
            if [[ -n "$BEST_FILE" ]]; then
                _add_dataset "lcc_${LANG}_ctx_${BEST_LOW}_${B}"
            fi
        done
        shopt -u nullglob
    done
}

if [[ -n "$DATASET_OVERRIDE" ]]; then
    read -r -a _REQ <<<"$DATASET_OVERRIDE"
    WANT_ALL=0
    WANT_CCEVAL=0
    WANT_LCC=0
    for d in "${_REQ[@]}"; do
        dl="$(echo "$d" | tr '[:upper:]' '[:lower:]')"
        if [[ "$dl" == "all" ]]; then
            WANT_ALL=1
        elif [[ "$dl" == "cceval" ]]; then
            WANT_CCEVAL=1
        elif [[ "$dl" == "lcc" ]]; then
            WANT_LCC=1
        fi
    done

    if [[ "$WANT_ALL" -eq 1 ]]; then
        _discover_cceval
        _discover_lcc
    else
        [[ "$WANT_CCEVAL" -eq 1 ]] && _discover_cceval
        [[ "$WANT_LCC" -eq 1 ]] && _discover_lcc
    fi

    # Also allow explicit dataset keys mixed with group keywords.
    for d in "${_REQ[@]}"; do
        dl="$(echo "$d" | tr '[:upper:]' '[:lower:]')"
        if [[ "$dl" == "all" || "$dl" == "cceval" || "$dl" == "lcc" ]]; then
            continue
        fi
        _add_dataset "$d"
    done
else
    _discover_cceval
    _discover_lcc
fi

# Convert GPU string to array
GPU_ARRAY=($GPUS)
NUM_GPUS=${#GPU_ARRAY[@]}
NUM_DATASETS=${#DATASETS[@]}

if [ $NUM_DATASETS -eq 0 ]; then
    echo "ERROR: No datasets found."
    echo "  CCEval dir: $CCEVAL_PROMPT_DIR"
    echo "  LCC dir:    $LCC_PROMPT_DIR"
    echo "  Budgets:    $BUDGETS"
    exit 1
fi

echo "=========================================="
echo "Parallel Multi-Dataset Benchmark"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Datasets:     ${DATASETS[*]}"
echo "  Max samples:  $MAX_SAMPLES"
echo "  Model:        $MODEL"
echo "  Prompt tok:   $PROMPT_TOKENIZER"
echo "  Tok suffix:   $TOKENIZER_SUFFIX"
echo "  Output dir:   $OUTPUT_DIR"
echo "  CCEval dir:   $CCEVAL_PROMPT_DIR"
echo "  LCC dir:      $LCC_PROMPT_DIR"
echo "  Budgets:      $BUDGETS"
echo "  GPUs:         ${GPU_ARRAY[*]} ($NUM_GPUS GPUs)"
echo "  Skip warmup:  $([ -n "$NO_WARMUP" ] && echo "Yes" || echo "No")"
echo "  Logs dir:     $LOG_DIR"
echo ""

# Check if we have enough GPUs
if [ $NUM_GPUS -lt $NUM_DATASETS ]; then
    echo "Warning: Only $NUM_GPUS GPUs available for $NUM_DATASETS datasets."
    echo "    Some datasets will run sequentially."
    echo ""
fi

echo "=========================================="
echo ""

# Create directories
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

echo "Starting parallel benchmark..."
echo "Logs will be saved to: $LOG_DIR"
echo ""

# Track failures robustly
FAILED=0

# Track GPU availability dynamically so finished jobs release devices before the next launch.
declare -a FREE_GPUS=("${GPU_ARRAY[@]}")
declare -a PIDS=()
declare -A PID_TO_GPU=()
declare -A PID_TO_DATASET=()
declare -A PID_TO_LOG=()

remove_pid() {
    local target="$1"
    local new_pids=()
    local p
    for p in "${PIDS[@]}"; do
        if [[ "$p" != "$target" ]]; then
            new_pids+=("$p")
        fi
    done
    PIDS=("${new_pids[@]}")
}

handle_finished() {
    local pid="$1"
    local rc="$2"
    local gpu="${PID_TO_GPU[$pid]:-unknown}"
    local dataset="${PID_TO_DATASET[$pid]:-unknown}"
    local log="${PID_TO_LOG[$pid]:-}"

    remove_pid "$pid"
    unset "PID_TO_GPU[$pid]" "PID_TO_DATASET[$pid]" "PID_TO_LOG[$pid]"
    FREE_GPUS+=("$gpu")

    if [[ "$rc" -ne 0 ]]; then
        echo "!!! FAILED $dataset on GPU $gpu (exit=$rc) (see $log)"
        return 1
    fi
    echo "<<< Done $dataset on GPU $gpu"
    return 0
}

wait_for_any_job() {
    local finished_pid=""
    local rc=0

    if wait -n -p finished_pid 2>/dev/null; then
        rc=0
    else
        rc=$?
    fi
    if [[ -n "$finished_pid" ]]; then
        handle_finished "$finished_pid" "$rc"
        return $?
    fi

    # Fallback for shells without `wait -n -p`
    local pid
    while true; do
        for pid in "${PIDS[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                if wait "$pid"; then
                    rc=0
                else
                    rc=$?
                fi
                handle_finished "$pid" "$rc"
                return $?
            fi
        done
        sleep 1
    done
}

cleanup() {
    if [[ ${#PIDS[@]} -gt 0 ]]; then
        local pid
        for pid in "${PIDS[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
    fi
}
trap cleanup INT TERM

launch_one() {
    local dataset="$1"
    local gpu_id="$2"
    local log_file="$3"

    local dataset_out_dir="${OUTPUT_DIR}/${dataset}"
    mkdir -p "$dataset_out_dir"

    echo ">>> Launching $dataset on GPU $gpu_id"
    echo "    Log:    $log_file"
    echo "    Output: $dataset_out_dir"

    (
        set -x
        EXTRA_ARGS=()
        if [[ -n "$NO_WARMUP" ]]; then
            EXTRA_ARGS+=(--no_warmup)
        fi
        CUDA_VISIBLE_DEVICES="$gpu_id" python "${ROOT_DIR}/evaluation/benchmark_multi_dataset.py" \
            --datasets "$dataset" \
            --max_samples "$MAX_SAMPLES" \
            --cceval_prompt_dir "$CCEVAL_PROMPT_DIR" \
            --lcc_prompt_dir "$LCC_PROMPT_DIR" \
            --budgets "${BUDGETS// /,}" \
            --model_name "$MODEL" \
            --output_dir "$dataset_out_dir" \
            "${EXTRA_ARGS[@]}"
    ) >"$log_file" 2>&1 &

    local pid=$!
    PIDS+=("$pid")
    PID_TO_GPU["$pid"]="$gpu_id"
    PID_TO_DATASET["$pid"]="$dataset"
    PID_TO_LOG["$pid"]="$log_file"
    echo "    PID:    $pid"
}

for dataset in "${DATASETS[@]}"; do
    while [[ ${#FREE_GPUS[@]} -eq 0 ]]; do
        if wait_for_any_job; then
            :
        else
            FAILED=$((FAILED + 1))
        fi
    done

    gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    log_file="$LOG_DIR/${dataset}_gpu${gpu}.log"
    launch_one "$dataset" "$gpu" "$log_file"
    sleep 1
done

echo ""
echo "=========================================="
echo "All $NUM_DATASETS datasets launched!"
echo "=========================================="
echo ""
echo "Waiting for all benchmarks to complete..."
echo "(You can monitor progress with: tail -f $LOG_DIR/*.log)"
echo ""

# Wait for remaining jobs
while [[ ${#PIDS[@]} -gt 0 ]]; do
    if wait_for_any_job; then
        :
    else
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=========================================="
if [ $FAILED -eq 0 ]; then
    echo "All benchmarks completed successfully."
else
    echo "Some benchmarks failed. Check logs for details."
fi
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Logs saved to: $LOG_DIR"
echo ""

# Show summary of results
echo "Quick summary:"
echo ""
for DATASET in "${DATASETS[@]}"; do
    RESULT_FILE="$OUTPUT_DIR/$DATASET/multi_dataset_benchmark_*.json"
    echo "  - $DATASET: Check $RESULT_FILE"
done
echo ""

exit $FAILED
