#!/usr/bin/env python3
"""
Create small test subsets from `data/_filtered_ts`.

Requirement:
  - Take the first N (default: 300) samples from each dataset JSONL.
  - For LCC datasets, only use the *test* split (skip validation).
  - Write the sampled JSONLs into a new output directory.

Typical inputs (produced by `data/filter_parseable.py`):
  - LCC_python_test_parseable.jsonl
  - LCC_java_test_parseable.jsonl
  - LCC_csharp_test_parseable.jsonl
  - cceval_python_test_parseable.jsonl
  - cceval_java_test_parseable.jsonl
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple


def _iter_jsonl_paths(in_dir: str) -> List[str]:
    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"Input directory not found: {in_dir}")
    paths: List[str] = []
    for name in sorted(os.listdir(in_dir)):
        if name.endswith(".jsonl"):
            paths.append(os.path.join(in_dir, name))
    return paths


def _should_include(name: str) -> bool:
    # LCC: only keep test split
    if name.startswith("LCC_") and "_validation_" in name:
        return False
    if name.startswith("LCC_") and "_validation" in name:
        return False
    return True


def _copy_first_n_lines(src: str, dst: str, n: int) -> Tuple[int, int]:
    total = 0
    kept = 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            if kept < n:
                fout.write(line)
                kept += 1
            if kept >= n:
                break
    return kept, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="data/_filtered_ts", help="Input directory containing JSONL files")
    parser.add_argument("--out_dir", type=str, default="data/_filtered_ts_mini300", help="Output directory for sampled JSONL files")
    parser.add_argument("-n", "--num_samples", type=int, default=300, help="Number of samples to keep from each JSONL")
    parser.add_argument("--strict", action="store_true", help="Fail if no eligible JSONL files are found")
    args = parser.parse_args()

    jsonl_paths = _iter_jsonl_paths(args.in_dir)
    eligible = [p for p in jsonl_paths if _should_include(os.path.basename(p))]

    if not eligible:
        msg = f"No eligible JSONL files found in {args.in_dir}"
        if args.strict:
            raise RuntimeError(msg)
        print(msg)
        return

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Sampling first {args.num_samples} lines per dataset:")
    for src in eligible:
        name = os.path.basename(src)
        dst = os.path.join(args.out_dir, name)
        kept, _total_seen = _copy_first_n_lines(src, dst, args.num_samples)
        print(f"  - {name}: wrote {kept} -> {dst}")


if __name__ == "__main__":
    main()

