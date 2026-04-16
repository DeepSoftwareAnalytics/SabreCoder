#!/usr/bin/env python3
"""
Build CCEval prompts with RAG-style crossfile context.

Prompt = crossfile_context + infile_left_context

Crossfile context is constructed by taking retrieved segments (ranked by BM25 score)
until the crossfile token budget is reached (2k/4k/8k/16k/32k by default).

Tokenizer is configurable via `--tokenizer` (HuggingFace `AutoTokenizer`).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import pyarrow.dataset as ds
from tqdm import tqdm
from transformers import AutoTokenizer


def _iter_parquet_rows(parquet_path: str, columns: Optional[List[str]] = None, batch_size: int = 256) -> Iterable[Dict]:
    dataset = ds.dataset(parquet_path, format="parquet")
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)
    for batch in scanner.to_batches():
        table = batch.to_pydict()
        n = len(next(iter(table.values()))) if table else 0
        for i in range(n):
            yield {k: table[k][i] for k in table.keys()}


def _load_retrieval_results(path: str) -> Dict[str, List[Dict]]:
    """
    Returns task_id -> list of result dicts (sorted by score desc in file).
    """
    by_task: Dict[str, List[Dict]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            task_id = obj["task_id"]
            by_task[task_id] = obj.get("results", [])
    return by_task


def _comment_prefix(language: str) -> str:
    return "# " if language == "python" else "// "


def _format_segment_block(language: str, seg: Dict) -> str:
    """
    Format one retrieved segment into commented text.
    """
    prefix = _comment_prefix(language)
    file_path = seg.get("file_path", "")
    start_line = seg.get("start_line", 0)
    end_line = seg.get("end_line", 0)
    text = seg.get("text", "") or ""

    lines = text.split("\n")
    out_lines: List[str] = []
    out_lines.append(f"{prefix}{file_path}: {start_line} - {end_line}\n")
    for ln in lines:
        out_lines.append(prefix + ln + "\n")
    out_lines.append("\n")
    return "".join(out_lines)


def _num_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _append_until_budget(
    tokenizer,
    language: str,
    ranked_segments: List[Dict],
    budget_tokens: int,
    exclude_file_path: Optional[str] = None,
) -> Tuple[str, int, int]:
    """
    Returns (crossfile_text, used_tokens, used_segments).

    Strategy:
      - Try to add whole segments in ranked order
      - If a segment would overflow, do a line-wise partial add for that segment then stop
    """
    prefix = _comment_prefix(language)
    out_parts: List[str] = []
    used_tokens = 0
    used_segments = 0

    for seg in ranked_segments:
        if exclude_file_path and (seg.get("file_path") == exclude_file_path):
            continue
        block = _format_segment_block(language, seg)
        block_tokens = _num_tokens(tokenizer, block)
        if used_tokens + block_tokens <= budget_tokens:
            out_parts.append(block)
            used_tokens += block_tokens
            used_segments += 1
            continue

        # Partial fill this segment (keep header, then add commented lines until budget)
        file_path = seg.get("file_path", "")
        start_line = seg.get("start_line", 0)
        end_line = seg.get("end_line", 0)
        header = f"{prefix}{file_path}: {start_line} - {end_line}\n"
        header_tokens = _num_tokens(tokenizer, header)
        if used_tokens + header_tokens > budget_tokens:
            break

        tmp_parts = [header]
        tmp_used = header_tokens
        for ln in (seg.get("text", "") or "").split("\n"):
            line = prefix + ln + "\n"
            t = _num_tokens(tokenizer, line)
            if used_tokens + tmp_used + t > budget_tokens:
                break
            tmp_parts.append(line)
            tmp_used += t

        if len(tmp_parts) > 1:
            tmp_parts.append("\n")
            out_parts.append("".join(tmp_parts))
            used_tokens += tmp_used
            used_segments += 1
        break

    return "".join(out_parts), used_tokens, used_segments


def build_prompts(
    parquet_path: str,
    language: str,
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
    # We only need token counting; avoid noisy warnings for long contexts.
    tokenizer.model_max_length = int(1e9)

    by_task = _load_retrieval_results(retrieval_jsonl)

    cols = ["task_id", "left_context", "right_context", "groundtruth", "path"]
    rows = _iter_parquet_rows(parquet_path, columns=cols)

    out_paths: List[str] = []
    writers = {}
    for b in budgets:
        out_path = os.path.join(out_dir, f"cceval_{language}_rag_{b}.jsonl")
        out_paths.append(out_path)
        writers[b] = open(out_path, "w", encoding="utf-8")

    try:
        written = 0
        for row in tqdm(rows, desc=f"Build prompts ({language})"):
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
                    tokenizer=tokenizer,
                    language=language,
                    ranked_segments=ranked,
                    budget_tokens=b,
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
                                "language": language,
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
    parser.add_argument("--parquet", type=str, required=True)
    parser.add_argument("--language", type=str, required=True, choices=["python", "java"])
    parser.add_argument("--retrieval_jsonl", type=str, required=True, help="BM25 retrieval output JSONL")
    parser.add_argument("--out_dir", type=str, default="data/_cceval_rag_prompts")
    parser.add_argument("--budgets", type=str, default="2048,4096,6144,8192,10240,12288,14336,16384")
    parser.add_argument("--tokenizer", type=str, default="deepseek-ai/deepseek-coder-1.3b-base")
    parser.add_argument("--allow_download", action="store_true", help="Allow HF download if tokenizer not cached")
    parser.add_argument("--max_tasks", type=int, default=None)
    args = parser.parse_args()

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    out_paths = build_prompts(
        parquet_path=args.parquet,
        language=args.language,
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
