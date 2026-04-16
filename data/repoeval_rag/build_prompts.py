#!/usr/bin/env python3
"""
Build RepoEval prompts with RAG-style crossfile context.

Prompt = crossfile_context (commented segments) + infile left_context

Crossfile context is constructed by taking retrieved segments (ranked by BM25 score)
until the crossfile token budget is reached (2k/4k/8k/16k/32k by default).

Tokenizer: DeepSeek Coder 1.3B Base (HF tokenizer).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import pyarrow.dataset as ds
from tqdm import tqdm
from transformers import AutoTokenizer


LANGUAGE = "python"


def _iter_parquet_rows(parquet_paths: List[str], columns: Optional[List[str]] = None, batch_size: int = 256) -> Iterable[Dict]:
    dataset = ds.dataset(parquet_paths, format="parquet")
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)
    for batch in scanner.to_batches():
        table = batch.to_pydict()
        n = len(next(iter(table.values()))) if table else 0
        for i in range(n):
            yield {k: table[k][i] for k in table.keys()}


def _load_retrieval_results(path: str) -> Dict[str, List[Dict]]:
    by_task: Dict[str, List[Dict]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            by_task[obj["task_id"]] = obj.get("results", [])
    return by_task


def _format_segment_block(seg: Dict) -> str:
    file_path = seg.get("file_path", "")
    start_line = seg.get("start_line", 0)
    end_line = seg.get("end_line", 0)
    text = seg.get("text", "") or ""

    out_lines: List[str] = []
    out_lines.append(f"# {file_path}: {start_line} - {end_line}\n")
    for ln in text.split("\n"):
        out_lines.append("# " + ln + "\n")
    out_lines.append("\n")
    return "".join(out_lines)


def _num_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _append_until_budget(
    tokenizer,
    ranked_segments: List[Dict],
    budget_tokens: int,
    exclude_file_path: Optional[str] = None,
) -> Tuple[str, int, int]:
    out_parts: List[str] = []
    used_tokens = 0
    used_segments = 0

    for seg in ranked_segments:
        # Prevent leakage: never retrieve segments from the current file of this task.
        # (Repo-level cache may contain this file because it can be a crossfile for other tasks.)
        if exclude_file_path and seg.get("file_path") == exclude_file_path:
            continue
        block = _format_segment_block(seg)
        block_tokens = _num_tokens(tokenizer, block)
        if used_tokens + block_tokens <= budget_tokens:
            out_parts.append(block)
            used_tokens += block_tokens
            used_segments += 1
            continue

        # Partial fill this segment (header + commented lines)
        header = f"# {seg.get('file_path','')}: {seg.get('start_line',0)} - {seg.get('end_line',0)}\n"
        header_tokens = _num_tokens(tokenizer, header)
        if used_tokens + header_tokens > budget_tokens:
            break

        tmp = [header]
        tmp_used = header_tokens
        for ln in (seg.get("text", "") or "").split("\n"):
            line = "# " + ln + "\n"
            t = _num_tokens(tokenizer, line)
            if used_tokens + tmp_used + t > budget_tokens:
                break
            tmp.append(line)
            tmp_used += t

        if len(tmp) > 1:
            tmp.append("\n")
            out_parts.append("".join(tmp))
            used_tokens += tmp_used
            used_segments += 1
        break

    return "".join(out_parts), used_tokens, used_segments


def build_prompts(
    parquet_paths: List[str],
    level: str,
    retrieval_jsonl: str,
    out_dir: str,
    budgets: List[int],
    tokenizer_name_or_path: str = "deepseek-ai/deepseek-coder-1.3b-base",
    allow_download: bool = False,
    max_tasks: Optional[int] = None,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=True,
        local_files_only=not allow_download,
    )

    by_task = _load_retrieval_results(retrieval_jsonl)

    cols = ["task_id", "left_context", "right_context", "groundtruth", "path"]
    rows = _iter_parquet_rows(parquet_paths, columns=cols)

    out_paths: List[str] = []
    writers = {}
    for b in budgets:
        out_path = os.path.join(out_dir, f"repoeval_{level}_rag_{b}.jsonl")
        out_paths.append(out_path)
        writers[b] = open(out_path, "w", encoding="utf-8")

    try:
        written = 0
        for row in tqdm(rows, desc=f"Build prompts ({level})"):
            task_id = row["task_id"]
            left = row.get("left_context") or ""
            right = row.get("right_context") or ""
            gt = row.get("groundtruth") or ""
            file_path = row.get("path") or ""

            ranked = by_task.get(task_id)
            if ranked is None:
                raise KeyError(f"Missing retrieval results for task_id={task_id}")

            for b in budgets:
                crossfile_text, used_toks, used_segs = _append_until_budget(
                    tokenizer,
                    ranked,
                    b,
                    exclude_file_path=file_path,
                )
                context = (crossfile_text + left) if crossfile_text else left
                writers[b].write(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "context": context,
                            "crossfile_context": crossfile_text,
                            "current_file_context": left,
                            "ground_truth": gt,
                            "right_context": right,
                            "metadata": {
                                "dataset": "repoeval",
                                "language": LANGUAGE,
                                "level": level,
                                "file_path": file_path,
                                "crossfile_budget_tokens": b,
                                "crossfile_used_tokens": used_toks,
                                "crossfile_used_segments": used_segs,
                                "tokenizer": tokenizer_name_or_path,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            written += 1
            if max_tasks is not None and written >= max_tasks:
                break
    finally:
        for f in writers.values():
            f.close()

    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, required=True, choices=["line_level", "api_level", "func_level"])
    parser.add_argument("--parquets", type=str, default=None)
    parser.add_argument("--retrieval_jsonl", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/_repoeval_rag_prompts")
    parser.add_argument("--budgets", type=str, default="2048,4096,8192,16384,32768")
    parser.add_argument("--tokenizer", type=str, default="deepseek-ai/deepseek-coder-1.3b-base")
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--max_tasks", type=int, default=None)
    args = parser.parse_args()

    if args.parquets:
        parquet_paths = [p.strip() for p in args.parquets.split(",") if p.strip()]
    else:
        base = os.path.join("data", "repoeval", args.level)
        parquet_paths = sorted(os.path.join(base, f) for f in os.listdir(base) if f.endswith(".parquet"))

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    out_paths = build_prompts(
        parquet_paths=parquet_paths,
        level=args.level,
        retrieval_jsonl=args.retrieval_jsonl,
        out_dir=args.out_dir,
        budgets=budgets,
        tokenizer_name_or_path=args.tokenizer,
        allow_download=args.allow_download,
        max_tasks=args.max_tasks,
    )
    print("Done. Wrote:")
    for p in out_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
