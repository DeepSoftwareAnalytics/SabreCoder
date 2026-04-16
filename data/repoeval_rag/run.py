#!/usr/bin/env python3
"""
One-click RepoEval RAG pipeline:
  1) Build repo-level segment cache per level (if missing)
  2) BM25 retrieval (groundtruth query) per task
  3) Build prompts with crossfile token budgets (DeepSeek Coder tokenizer)
"""

from __future__ import annotations

import argparse
import os

from build_library import build_repo_library, LANGUAGE as LANG
from retrieve_bm25 import retrieve
from build_prompts import build_prompts


def _level_manifest(cache_dir: str, level: str) -> str:
    return os.path.join(cache_dir, LANG, level, "manifest.jsonl")


def _parquet_paths_for_level(level: str) -> list[str]:
    base = os.path.join("data", "repoeval", level)
    return sorted(os.path.join(base, f) for f in os.listdir(base) if f.endswith(".parquet"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="data/_repoeval_rag_cache")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--levels", type=str, default="line_level,api_level,func_level")
    parser.add_argument("--rebuild", action="store_true")

    parser.add_argument("--build_workers", type=int, default=8)
    parser.add_argument("--retrieve_workers", type=int, default=8)
    parser.add_argument("--topk", type=int, default=200)

    parser.add_argument("--prompt_budgets", type=str, default="2048,4096,8192,16384,32768")
    parser.add_argument("--tokenizer", type=str, default="deepseek-ai/deepseek-coder-1.3b-base")
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--skip_prompt_build", action="store_true")

    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--max_crossfile_files_per_task", type=int, default=None)
    args = parser.parse_args()

    budgets = [int(x.strip()) for x in args.prompt_budgets.split(",") if x.strip()]
    levels = [x.strip() for x in args.levels.split(",") if x.strip()]

    for level in levels:
        parquets = _parquet_paths_for_level(level)
        manifest = _level_manifest(args.cache_dir, level)
        need_build = args.rebuild or (not os.path.exists(manifest))
        if need_build:
            stats = build_repo_library(
                parquet_paths=parquets,
                level=level,
                cache_dir=args.cache_dir,
                workers=args.build_workers,
                max_tasks=args.max_tasks,
                max_crossfile_files_per_task=args.max_crossfile_files_per_task,
            )
            print(f"[{level}] cache built: repos={stats.repos} tasks={stats.tasks} files={stats.files} segments={stats.segments}")
        else:
            print(f"[{level}] cache exists: {manifest}")

        retrieval_out = os.path.join(args.out_dir, f"_repoeval_rag_results_{level}.jsonl")
        retrieve(
            parquet_paths=parquets,
            level=level,
            cache_dir=args.cache_dir,
            out_path=retrieval_out,
            topk=args.topk,
            workers=args.retrieve_workers,
            max_tasks=args.max_tasks,
        )
        print(f"[{level}] retrieval done: {retrieval_out}")

        if not args.skip_prompt_build:
            prompt_out_dir = os.path.join(args.out_dir, "_repoeval_rag_prompts")
            out_paths = build_prompts(
                parquet_paths=parquets,
                level=level,
                retrieval_jsonl=retrieval_out,
                out_dir=prompt_out_dir,
                budgets=budgets,
                tokenizer_name_or_path=args.tokenizer,
                allow_download=args.allow_download,
                max_tasks=args.max_tasks,
            )
            print(f"[{level}] prompts built ({len(out_paths)} budgets) in {prompt_out_dir}")


if __name__ == "__main__":
    main()

