#!/usr/bin/env python3
"""
Retrieve crossfile segments for each CCEval task using BM25 (rank_bm25).

Query: groundtruth (from parquet 'groundtruth')
Corpus: per-task segments cache produced by `build_library.py`

Parallelism:
  - Use multiprocessing across tasks with `--workers` (processes).

Outputs:
  out_path (JSONL), one line per task:
    {
      "task_id": ...,
      "language": ...,
      "query": groundtruth,
      "results": [
        {"score": float, "file_path": str, "start_line": int, "end_line": int, "text": str, "segment_kind": str, "node_type": str|null},
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import pyarrow.dataset as ds
from rank_bm25 import BM25Okapi
from tqdm import tqdm


_TOKEN_RE = re.compile(r"[_A-Za-z][_A-Za-z0-9]*|\\d+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _task_filename(task_id: str) -> str:
    safe = task_id.replace(os.sep, "__").replace("/", "__")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", safe)
    return safe + ".jsonl"


def _iter_parquet_rows(parquet_path: str, columns: Optional[List[str]] = None, batch_size: int = 256) -> Iterable[Dict]:
    dataset = ds.dataset(parquet_path, format="parquet")
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)
    for batch in scanner.to_batches():
        table = batch.to_pydict()
        n = len(next(iter(table.values()))) if table else 0
        for i in range(n):
            yield {k: table[k][i] for k in table.keys()}


def _load_task_segments(task_file: str) -> List[Dict]:
    segs: List[Dict] = []
    with open(task_file, "r", encoding="utf-8") as f:
        for line in f:
            segs.append(json.loads(line))
    return segs


def _rank_all(scores) -> List[int]:
    # Descending sort by score.
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


def _retrieve_one(args: Tuple[int, str, str, str, Optional[int]]) -> Tuple[int, Dict]:
    """
    Worker for one task.
    Returns (idx, output_obj) to preserve input order.
    """
    idx, task_id, query, tasks_dir, topk = args
    task_file = os.path.join(tasks_dir, _task_filename(task_id))
    if not os.path.exists(task_file):
        raise FileNotFoundError(f"Missing cached task file: {task_file}")

    segs = _load_task_segments(task_file)
    corpus_tokens = [_tokenize(s.get("text") or "") for s in segs]
    bm25 = BM25Okapi(corpus_tokens)

    q_tokens = _tokenize(query)
    scores = bm25.get_scores(q_tokens)
    order = _rank_all(scores)
    if topk is not None:
        order = order[:topk]

    results = []
    for doc_idx in order:
        s = segs[doc_idx]
        results.append(
            {
                "score": float(scores[doc_idx]),
                "file_path": s.get("file_path", ""),
                "start_line": s.get("start_line", 0),
                "end_line": s.get("end_line", 0),
                "segment_kind": s.get("segment_kind", ""),
                "node_type": s.get("node_type", None),
                "text": s.get("text", ""),
            }
        )

    out_obj = {
        "task_id": task_id,
        "query": query,
        "results": results,
    }
    return idx, out_obj


def retrieve(
    parquet_path: str,
    language: str,
    cache_dir: str,
    out_path: str,
    topk: Optional[int] = None,
    workers: int = 1,
    max_tasks: Optional[int] = None,
) -> None:
    tasks_dir = os.path.join(cache_dir, language, "tasks")
    if not os.path.isdir(tasks_dir):
        raise FileNotFoundError(f"RAG cache not found: {tasks_dir} (run build first)")

    cols = ["task_id", "groundtruth"]
    rows: List[Tuple[int, str, str, str, Optional[int]]] = []
    for i, row in enumerate(_iter_parquet_rows(parquet_path, columns=cols)):
        task_id = row["task_id"]
        query = row.get("groundtruth") or ""
        rows.append((i, task_id, query, tasks_dir, topk))
        if max_tasks is not None and len(rows) >= max_tasks:
            break

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    results_by_idx: List[Optional[Dict]] = [None] * len(rows)

    if workers == 1:
        for item in tqdm(rows, desc=f"BM25 retrieve ({language})"):
            idx, out_obj = _retrieve_one(item)
            results_by_idx[idx] = out_obj
    else:
        with mp.Pool(processes=workers) as pool:
            for idx, out_obj in tqdm(
                pool.imap_unordered(_retrieve_one, rows, chunksize=8),
                total=len(rows),
                desc=f"BM25 retrieve ({language}, workers={workers})",
            ):
                results_by_idx[idx] = out_obj

    with open(out_path, "w", encoding="utf-8") as out:
        for out_obj in results_by_idx:
            assert out_obj is not None
            out.write(json.dumps({"language": language, **out_obj}, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, required=True)
    parser.add_argument("--language", type=str, required=True, choices=["python", "java"])
    parser.add_argument("--cache_dir", type=str, default="data/_cceval_rag_cache")
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Processes for retrieval (BM25 scoring per task)")
    parser.add_argument("--max_tasks", type=int, default=None)
    args = parser.parse_args()

    out_path = args.out_path or f"data/_cceval_rag_results_{args.language}.jsonl"
    retrieve(
        parquet_path=args.parquet,
        language=args.language,
        cache_dir=args.cache_dir,
        out_path=out_path,
        topk=args.topk,
        workers=args.workers,
        max_tasks=args.max_tasks,
    )
    print(f"Done. Wrote results to {out_path}")


if __name__ == "__main__":
    main()

