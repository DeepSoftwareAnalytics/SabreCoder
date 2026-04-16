#!/usr/bin/env python3
"""
Build LCC prompt subsets by context-length token budgets.

LCC has no crossfile context, so we bucket samples by the number of tokens in `context`.

Default buckets (low, high] in tokens:
  - 1k-2k:  1024 < tokens <= 2048
  - 2k-4k:  2048 < tokens <= 4096
  - 4k-6k:  4096 < tokens <= 6144
  - 6k-8k:  6144 < tokens <= 8192
  - 8k-10k: 8192 < tokens <= 10240
  - 10k-12k: 10240 < tokens <= 12288
  - 12k-14k: 12288 < tokens <= 14336
  - 14k-16k: 14336 < tokens <= 16384

Tokenizer is configurable via `--tokenizer`/`--tokenizers` (HF `AutoTokenizer`).

Inputs (default): JSONL files under data/_filtered_ts:
  - LCC_python_test_parseable.jsonl
  - LCC_java_test_parseable.jsonl
  - LCC_csharp_test_parseable.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer


DEFAULT_BUDGETS = [2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384]


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


def _iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _bucket_ranges(budgets: List[int]) -> List[Tuple[int, int]]:
    # Produce contiguous (low, high] ranges from budget highs.
    #
    # We interpret budgets as *upper bounds* and derive buckets by:
    #   - First bucket: (budgets[0]//2, budgets[0]]
    #   - Next buckets: (prev_high, high]
    #
    # This keeps contiguous bucket boundaries while also supporting
    # finer-grained budgets like 4k-6k and 6k-8k.
    budgets = sorted({int(x) for x in budgets})
    if not budgets:
        return []
    ranges: List[Tuple[int, int]] = []
    low = budgets[0] // 2
    for high in budgets:
        ranges.append((low, high))
        low = high
    return ranges


def _bucket_for_tokens(n_tokens: int, ranges: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    for low, high in ranges:
        if low < n_tokens <= high:
            return (low, high)
    return None


def _default_inputs(in_dir: str, splits: List[str]) -> List[Tuple[str, str, str]]:
    """
    Returns list of (dataset_name, split, path).
    """
    datasets = ["LCC_python", "LCC_java", "LCC_csharp"]
    out: List[Tuple[str, str, str]] = []
    for ds in datasets:
        for split in splits:
            fn = f"{ds}_{split}_parseable.jsonl"
            p = os.path.join(in_dir, fn)
            if os.path.exists(p):
                out.append((ds, split, p))
    return out


def build_buckets_for_file(
    tokenizer,
    tokenizer_name_or_path: str,
    in_path: str,
    dataset_name: str,
    split: str,
    out_dir: str,
    budgets: List[int],
    max_samples: Optional[int] = None,
) -> Dict[str, int]:
    os.makedirs(out_dir, exist_ok=True)
    ranges = _bucket_ranges(budgets)

    writers: Dict[Tuple[int, int], object] = {}
    counts: Dict[str, int] = {"total": 0}
    for low, high in ranges:
        out_path = os.path.join(out_dir, f"{dataset_name}_{split}_ctx_{low}_{high}.jsonl")
        writers[(low, high)] = open(out_path, "w", encoding="utf-8")
        counts[f"{low}_{high}"] = 0

    try:
        for obj in tqdm(_iter_jsonl(in_path), desc=f"Bucket {dataset_name}:{split}"):
            counts["total"] += 1
            if max_samples is not None and counts["total"] > max_samples:
                break

            context = obj.get("context", "")
            n_tokens = len(tokenizer.encode(context, add_special_tokens=False))
            br = _bucket_for_tokens(n_tokens, ranges)
            if br is None:
                continue

            low, high = br
            key = f"{low}_{high}"
            counts[key] += 1

            # Augment metadata deterministically.
            obj = dict(obj)
            meta = dict(obj.get("metadata") or {})
            meta["lcc_context_tokens"] = n_tokens
            meta["lcc_bucket_low"] = low
            meta["lcc_bucket_high"] = high
            meta["tokenizer"] = tokenizer_name_or_path
            obj["metadata"] = meta

            writers[(low, high)].write(json.dumps(obj, ensure_ascii=False) + "\n")
    finally:
        for f in writers.values():
            f.close()

    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="data/_filtered_ts")
    parser.add_argument("--out_dir", type=str, default="data/_lcc_budget_prompts")
    parser.add_argument("--splits", type=str, default="test", help="Comma-separated: test,validation")
    parser.add_argument("--budgets", type=str, default="2048,4096,6144,8192,10240,12288,14336,16384")
    parser.add_argument("--tokenizer", type=str, default="deepseek-ai/deepseek-coder-1.3b-base")
    parser.add_argument(
        "--tokenizers",
        type=str,
        default=None,
        help="Comma-separated tokenizer name/path list; when set, bucketing is run once per tokenizer into suffixed directories.",
    )
    parser.add_argument("--allow_download", action="store_true", help="Allow HF download if tokenizer not cached")
    parser.add_argument("--max_samples", type=int, default=None, help="Debug cap per dataset split")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    tokenizers = _parse_tokenizers(args.tokenizers, tokenizer=args.tokenizer)

    inputs = _default_inputs(args.in_dir, splits=splits)
    if not inputs:
        raise FileNotFoundError(f"No input JSONL files found under {args.in_dir} for splits={splits}")

    for tok in tokenizers:
        tokenizer = AutoTokenizer.from_pretrained(
            tok,
            trust_remote_code=True,
            local_files_only=not args.allow_download,
        )
        # We only need token counting; avoid noisy warnings for long contexts.
        tokenizer.model_max_length = int(1e9)

        out_dir = f"{args.out_dir}__{_slugify_tokenizer_name(tok)}" if len(tokenizers) > 1 else args.out_dir
        for ds_name, split, path in inputs:
            counts = build_buckets_for_file(
                tokenizer=tokenizer,
                tokenizer_name_or_path=tok,
                in_path=path,
                dataset_name=ds_name,
                split=split,
                out_dir=out_dir,
                budgets=budgets,
                max_samples=args.max_samples,
            )
            print(f"\n[{tok}] {ds_name}:{split}")
            print(f"  total_seen: {counts['total']}")
            for low, high in _bucket_ranges(budgets):
                key = f"{low}_{high}"
                print(f"  bucket_{low}_{high}: {counts.get(key, 0)}")


if __name__ == "__main__":
    main()
