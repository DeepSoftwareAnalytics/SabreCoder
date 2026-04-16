"""
Multi-dataset benchmark entrypoint for SabreCoder.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, SCRIPT_DIR)

from utils import print_results_table


def _resolve_output_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _discover_dataset_configs(
    cceval_prompt_dir: str,
    lcc_prompt_dir: str,
    budgets: list[int],
) -> dict[str, dict]:
    configs: dict[str, dict] = {}
    eval_dir = os.path.dirname(os.path.abspath(__file__))

    def _rel(path: str) -> str:
        return os.path.relpath(os.path.abspath(path), eval_dir)

    for lang in ["python", "java"]:
        for budget in budgets:
            filename = f"cceval_{lang}_rag_{budget}.jsonl"
            path = os.path.join(cceval_prompt_dir, filename)
            if os.path.exists(path):
                key = f"cceval_{lang}_rag_{budget}"
                configs[key] = {
                    "name": f"CCEval {lang} (RAG crossfile budget={budget})",
                    "path": _rel(path),
                    "language": lang,
                    "default_samples": 1000,
                    "max_length": int(budget),
                }

    budgets_set = {int(x) for x in budgets}
    lcc_dir = Path(lcc_prompt_dir)
    if lcc_dir.exists():
        for lang in ["python", "java", "csharp"]:
            pattern = re.compile(rf"^LCC_{lang}_test_ctx_(\d+)_(\d+)\.jsonl$")
            best_by_high: dict[int, tuple[int, Path]] = {}
            for path in lcc_dir.iterdir():
                match = pattern.match(path.name)
                if not match:
                    continue
                low = int(match.group(1))
                high = int(match.group(2))
                if high not in budgets_set:
                    continue
                prev = best_by_high.get(high)
                if prev is None or low > prev[0]:
                    best_by_high[high] = (low, path)
            for high, (low, path) in best_by_high.items():
                key = f"lcc_{lang}_ctx_{low}_{high}"
                configs[key] = {
                    "name": f"LCC {lang} (context tokens in ({low},{high}])",
                    "path": _rel(str(path)),
                    "language": lang,
                    "default_samples": 1000,
                    "max_length": int(high),
                }

    return configs


def _build_command(
    script_path: str,
    dataset_path: str,
    dataset_key: str,
    max_samples: int,
    effective_max_length: int,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        script_path,
        "--model_name",
        args.model_name,
        "--data_path",
        dataset_path,
        "--max_samples",
        str(max_samples),
        "--max_length",
        str(effective_max_length),
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

    cmd.append("--crossfile_full_attention" if args.crossfile_full_attention else "--no-crossfile_full_attention")

    if args.no_warmup:
        cmd.append("--no_warmup")
    if args.use_sparse_for_generation:
        cmd.append("--use_sparse_for_generation")
    if args.use_compile:
        cmd.extend(["--use_compile", "--compile_mode", args.compile_mode])
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

    if dataset_key.startswith("cceval_"):
        precomputed_blocks = dataset_path.replace(".jsonl", "_blocks.pt")
        full_blocks = os.path.join(os.path.dirname(os.path.abspath(__file__)), precomputed_blocks)
        if os.path.exists(full_blocks):
            cmd.extend(["--precomputed_blocks_path", precomputed_blocks])

    return cmd


def run_sabrecoder_on_dataset(
    dataset_config: dict,
    dataset_key: str,
    max_samples: int,
    args: argparse.Namespace,
) -> dict | None:
    dataset_name = dataset_config["name"]
    dataset_path = dataset_config["path"]
    effective_max_length = args.max_length
    if args.respect_dataset_max_length and "max_length" in dataset_config:
        effective_max_length = int(dataset_config["max_length"])

    print(f"\n{'=' * 80}")
    print(f"Running: SabreCoder on {dataset_name}")
    print(f"Max length: {effective_max_length}")
    print(f"{'=' * 80}\n")

    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_path)
    if not os.path.exists(full_path):
        print(f"WARNING: Dataset not found at {full_path}, skipping...")
        return None

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    script_path = os.path.join(project_root, "methods/sabrecoder/run_eval.py")

    try:
        subprocess.run(
            _build_command(script_path, dataset_path, dataset_key, max_samples, effective_max_length, args),
            check=True,
            capture_output=False,
            text=True,
        )

        dataset_output_name = (
            os.path.basename(dataset_path)
            .replace(".jsonl", "")
            .replace("_filtered", "")
            .replace("_longprompt", "")
        )
        result_path = os.path.join(args.output_dir, "sabrecoder", dataset_output_name, "results.json")
        if not os.path.exists(result_path):
            print(f"WARNING: Result file not found: {result_path}")
            return None

        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        result["method"] = "SabreCoder"
        result["dataset"] = dataset_name
        result["dataset_key"] = dataset_key
        return result
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: SabreCoder on {dataset_name} failed with exit code {exc.returncode}")
        return None
    except Exception as exc:
        print(f"ERROR: SabreCoder on {dataset_name} failed: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Multi-dataset benchmark for SabreCoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-coder-1.3b-base")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--respect_dataset_max_length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the dataset budget or upper bucket bound as max_length when available.",
    )
    parser.add_argument("--no_warmup", action="store_true")
    parser.add_argument("--use_sparse_for_generation", action="store_true")
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead")

    parser.add_argument("--cceval_prompt_dir", type=str, default="../data/_cceval_rag_prompts")
    parser.add_argument("--lcc_prompt_dir", type=str, default="../data/_lcc_budget_prompts")
    parser.add_argument(
        "--budgets",
        type=str,
        default="2048,4096,6144,8192,10240,12288,14336,16384",
    )
    parser.add_argument("--datasets", type=str, nargs="+", default=["all"])
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument(
        "--dataset_samples",
        type=str,
        nargs="*",
        help="Override samples with dataset:samples pairs.",
    )

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
    parser.add_argument("--use_token_level_sparsity", action="store_true", default=False)

    args = parser.parse_args()
    args.output_dir = _resolve_output_path(args.output_dir)
    if args.use_crossfile_chunk_similarity and not args.use_chunk_similarity:
        parser.error("--use_crossfile_chunk_similarity requires --use_chunk_similarity.")

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    eval_dir = os.path.dirname(os.path.abspath(__file__))

    cceval_prompt_dir = args.cceval_prompt_dir
    if not os.path.isabs(cceval_prompt_dir):
        cceval_prompt_dir = os.path.abspath(os.path.join(eval_dir, cceval_prompt_dir))
    lcc_prompt_dir = args.lcc_prompt_dir
    if not os.path.isabs(lcc_prompt_dir):
        lcc_prompt_dir = os.path.abspath(os.path.join(eval_dir, lcc_prompt_dir))

    dataset_configs = _discover_dataset_configs(
        cceval_prompt_dir=cceval_prompt_dir,
        lcc_prompt_dir=lcc_prompt_dir,
        budgets=budgets,
    )

    requested = [item.lower() for item in args.datasets]
    if "all" in requested:
        datasets_to_run = list(dataset_configs.keys())
    else:
        datasets_to_run = []
        for item in args.datasets:
            item_lower = item.lower()
            if item_lower == "cceval":
                datasets_to_run.extend([key for key in dataset_configs if key.startswith("cceval_")])
            elif item_lower == "lcc":
                datasets_to_run.extend([key for key in dataset_configs if key.startswith("lcc_")])
            else:
                datasets_to_run.append(item_lower)

    missing = [key for key in datasets_to_run if key not in dataset_configs]
    if missing:
        raise ValueError(f"Unknown dataset keys: {missing}. Discovered keys: {sorted(dataset_configs.keys())}")

    dataset_samples_map: dict[str, int] = {}
    if args.dataset_samples:
        for item in args.dataset_samples:
            if ":" not in item:
                continue
            dataset, samples = item.split(":", 1)
            dataset_samples_map[dataset] = int(samples)

    print("\n" + "=" * 80)
    print("MULTI-DATASET SABRECODER BENCHMARK")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {args.model_name}")
    print(f"Datasets: {', '.join(datasets_to_run)}")
    print(f"Default samples per dataset: {args.max_samples}")
    if dataset_samples_map:
        print(f"Custom sample counts: {dataset_samples_map}")
    print("=" * 80 + "\n")

    all_results: dict[str, list[dict]] = {}
    for dataset_key in datasets_to_run:
        dataset_config = dataset_configs[dataset_key]
        max_samples = dataset_samples_map.get(dataset_key, args.max_samples)

        print(f"\n{'#' * 80}")
        print(f"# DATASET: {dataset_config['name']} ({max_samples} samples)")
        print(f"{'#' * 80}\n")

        result = run_sabrecoder_on_dataset(dataset_config, dataset_key, max_samples, args)
        all_results[dataset_config["name"]] = [result] if result is not None else []
        if result is None:
            print("WARNING: SabreCoder did not produce results")

    print("\n" + "=" * 80)
    print("FINAL RESULTS BY DATASET")
    print("=" * 80)
    for dataset_name, results in all_results.items():
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 80}")
        if results:
            print_results_table(results)
        else:
            print("No results available for this dataset.")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_output = os.path.join(args.output_dir, f"multi_dataset_benchmark_{timestamp}.json")
    with open(combined_output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "configuration": {
                    "model": args.model_name,
                    "datasets": datasets_to_run,
                    "default_samples": args.max_samples,
                    "dataset_samples": dataset_samples_map,
                },
                "results": all_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nCombined results saved to: {combined_output}")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Multi-dataset benchmark complete!\n")


if __name__ == "__main__":
    main()
