#!/usr/bin/env python3
"""
One-click entry for CCEval RAG pipeline:

  1) Build per-task segment library (cache) if missing
  2) Run BM25 retrieval with groundtruth as query

Defaults:
  - Build mode: parse_once (each crossfile file parsed once)
  - Languages: python + java
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Optional

from build_library import build_library
from retrieve_bm25 import retrieve
from build_prompts import build_prompts


def _default_parquet(language: str) -> str:
    if language == "python":
        return "data/cceval/python/test.parquet"
    if language == "java":
        return "data/cceval/java/test.parquet"
    raise ValueError(language)


def _cache_manifest(cache_dir: str, language: str) -> str:
    return os.path.join(cache_dir, language, "manifest.jsonl")


def _slugify_tokenizer_name(tokenizer_name_or_path: str) -> str:
    s = tokenizer_name_or_path.strip().lower()
    s = s.replace("/", "--")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "tokenizer"


def _parse_tokenizers(tokenizers: Optional[str], tokenizer: str) -> List[str]:
    if tokenizers is None:
        return [tokenizer]
    out = [t.strip() for t in tokenizers.split(",") if t.strip()]
    if not out:
        raise ValueError("--tokenizers was provided but empty")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="data/_cceval_rag_cache")
    parser.add_argument("--out_dir", type=str, default="data", help="Directory for retrieval outputs")
    parser.add_argument(
        "--prompt_out_dir",
        type=str,
        default=None,
        help="Directory for prompt outputs (default: <out_dir>/_cceval_rag_prompts)",
    )
    parser.add_argument("--languages", type=str, default="python,java", help="Comma-separated: python,java")
    parser.add_argument("--mode", type=str, default="parse_once", choices=["parse_once", "incremental"])
    parser.add_argument("--max_lines", type=int, default=10, help="Max lines before forcing a segment (incremental)")
    parser.add_argument("--build_workers", type=int, default=8, help="Thread workers for build")
    parser.add_argument("--retrieve_workers", type=int, default=8, help="Process workers for retrieval")
    parser.add_argument("--topk", type=int, default=200, help="Top-k segments per task")
    parser.add_argument(
        "--prompt_budgets",
        type=str,
        default="2048,4096,6144,8192,10240,12288,14336,16384",
        help="Comma-separated crossfile token budgets",
    )
    parser.add_argument("--tokenizer", type=str, default="deepseek-ai/deepseek-coder-1.3b-base", help="Tokenizer name/path for budgeting")
    parser.add_argument(
        "--tokenizers",
        type=str,
        default=None,
        help="Comma-separated tokenizer name/path list; when set, prompts are built once per tokenizer into suffixed directories.",
    )
    parser.add_argument("--allow_download", action="store_true", help="Allow HF download if tokenizer not cached")
    parser.add_argument("--skip_prompt_build", action="store_true", help="Skip prompt building step")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--max_tasks", type=int, default=None, help="Debug cap for tasks (build+retrieve)")
    parser.add_argument("--max_crossfile_files_per_task", type=int, default=None, help="Debug cap for build")
    args = parser.parse_args()

    budgets = [int(x.strip()) for x in args.prompt_budgets.split(",") if x.strip()]
    languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    tokenizers = _parse_tokenizers(args.tokenizers, tokenizer=args.tokenizer)
    prompt_out_dir_base = args.prompt_out_dir or os.path.join(args.out_dir, "_cceval_rag_prompts")
    for lang in languages:
        parquet = _default_parquet(lang)
        manifest = _cache_manifest(args.cache_dir, lang)
        need_build = args.rebuild or (not os.path.exists(manifest))
        if need_build:
            stats = build_library(
                parquet_path=parquet,
                language=lang,
                out_dir=args.cache_dir,
                mode=args.mode,
                max_lines=args.max_lines,
                workers=args.build_workers,
                max_tasks=args.max_tasks,
                max_crossfile_files_per_task=args.max_crossfile_files_per_task,
            )
            print(f"[{lang}] cache built: tasks={stats.tasks} files={stats.files} segments={stats.segments}")
        else:
            print(f"[{lang}] cache exists: {manifest}")

        retrieval_out_path = os.path.join(args.out_dir, f"_cceval_rag_results_{lang}.jsonl")
        retrieve(
            parquet_path=parquet,
            language=lang,
            cache_dir=args.cache_dir,
            out_path=retrieval_out_path,
            topk=args.topk,
            workers=args.retrieve_workers,
            max_tasks=args.max_tasks,
        )
        print(f"[{lang}] retrieval done: {retrieval_out_path}")

        if not args.skip_prompt_build:
            for tok in tokenizers:
                prompt_out_dir = (
                    f"{prompt_out_dir_base}__{_slugify_tokenizer_name(tok)}" if len(tokenizers) > 1 else prompt_out_dir_base
                )
                out_paths = build_prompts(
                    parquet_path=parquet,
                    language=lang,
                    retrieval_jsonl=retrieval_out_path,
                    out_dir=prompt_out_dir,
                    budgets=budgets,
                    tokenizer_name_or_path=tok,
                    allow_download=args.allow_download,
                    max_tasks=args.max_tasks,
                )
                print(f"[{lang}] prompts built ({len(out_paths)} budgets) in {prompt_out_dir} (tokenizer={tok})")


if __name__ == "__main__":
    main()
