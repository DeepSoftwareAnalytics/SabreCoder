"""
Single-dataset benchmark entrypoint for SabreCoder.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, SCRIPT_DIR)

from utils import print_results_table


def _resolve_input_path(path: str) -> str:
    if os.path.isabs(path):
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


def _build_command(script_path: str, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        script_path,
        "--model_name",
        args.model_name,
        "--data_path",
        args.data_path,
        "--max_samples",
        str(args.max_samples),
        "--max_length",
        str(args.max_length),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--output_dir",
        args.output_dir,
        "--window_size",
        str(args.window_size),
        "--block_size",
        str(args.block_size),
        "--num_prefix_tokens",
        str(args.num_prefix_tokens),
        "--num_suffix_tokens",
        str(args.num_suffix_tokens),
        "--global_last_k_chunks",
        str(args.global_last_k_chunks),
        "--max_chunk_tokens",
        str(args.max_chunk_tokens),
        "--chunk_similarity_top_percent",
        str(args.chunk_similarity_top_percent),
        "--chunk_similarity_max_tokens_per_chunk",
        str(args.chunk_similarity_max_tokens_per_chunk),
        "--chunk_similarity_max_neighbors",
        str(args.chunk_similarity_max_neighbors),
        "--use_block_level_mask",
        str(args.use_block_level_mask),
    ]

    if args.no_warmup:
        cmd.append("--no_warmup")
    if args.use_sparse_for_generation:
        cmd.append("--use_sparse_for_generation")
    if args.use_compile:
        cmd.extend(["--use_compile", "--compile_mode", args.compile_mode])
    cmd.append("--crossfile_full_attention" if args.crossfile_full_attention else "--no-crossfile_full_attention")
    cmd.append("--use_chunk_similarity" if args.use_chunk_similarity else "--no-use_chunk_similarity")
    cmd.append(
        "--use_crossfile_chunk_similarity"
        if args.use_crossfile_chunk_similarity
        else "--no-use_crossfile_chunk_similarity"
    )
    if args.crossfile_chunk_similarity_top_percent is not None:
        cmd.extend(
            [
                "--crossfile_chunk_similarity_top_percent",
                str(args.crossfile_chunk_similarity_top_percent),
            ]
        )
    if args.use_token_level_sparsity:
        cmd.append("--use_token_level_sparsity")

    return cmd


def run_sabrecoder(args: argparse.Namespace) -> dict | None:
    script_path = os.path.join(PROJECT_ROOT, "methods/sabrecoder/run_eval.py")

    print(f"\n{'=' * 80}")
    print("Running: SabreCoder")
    print(f"{'=' * 80}\n")

    try:
        subprocess.run(_build_command(script_path, args), check=True, capture_output=False, text=True)

        dataset_name = (
            os.path.basename(args.data_path)
            .replace(".jsonl", "")
            .replace("_filtered", "")
            .replace("_longprompt", "")
        )
        result_file = os.path.join(args.output_dir, "sabrecoder", dataset_name, "results.json")
        if not os.path.exists(result_file):
            print(f"WARNING: Result file not found: {result_file}")
            return None

        with open(result_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: SabreCoder failed with error: {exc}")
        return None
    except Exception as exc:
        print(f"ERROR: Unexpected error running SabreCoder: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Single-dataset SabreCoder benchmark")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-coder-1.3b-base")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/_lcc_budget_prompts/LCC_python_test_ctx_12288_14336.jsonl",
    )
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--no_warmup", action="store_true")
    parser.add_argument("--use_sparse_for_generation", action="store_true")
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead")
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--num_prefix_tokens", type=int, default=4)
    parser.add_argument("--num_suffix_tokens", type=int, default=128)
    parser.add_argument("--global_last_k_chunks", type=int, default=0)
    parser.add_argument("--max_chunk_tokens", type=int, default=128)
    parser.add_argument(
        "--crossfile_full_attention",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--use_chunk_similarity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--chunk_similarity_top_percent", type=float, default=0.2)
    parser.add_argument("--chunk_similarity_max_tokens_per_chunk", type=int, default=128)
    parser.add_argument("--chunk_similarity_max_neighbors", type=int, default=8)
    parser.add_argument(
        "--use_crossfile_chunk_similarity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--crossfile_chunk_similarity_top_percent", type=float, default=0.2)
    parser.add_argument("--use_block_level_mask", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--use_token_level_sparsity", action="store_true")
    args = parser.parse_args()
    args.data_path = _resolve_input_path(args.data_path)
    args.output_dir = _resolve_output_path(args.output_dir)

    print("\n" + "=" * 80)
    print("SABRECODER BENCHMARK")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.data_path}")
    print(f"Samples: {args.max_samples}")
    print("=" * 80 + "\n")

    result = run_sabrecoder(args)
    if result is None:
        print("\nERROR: SabreCoder did not produce valid results.")
        return

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print_results_table([result])

    os.makedirs(args.output_dir, exist_ok=True)
    combined_output = os.path.join(args.output_dir, "benchmark_results.json")
    with open(combined_output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "configuration": vars(args),
                "results": [result],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nResults saved to: {combined_output}")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Benchmark complete!\n")


if __name__ == "__main__":
    main()
