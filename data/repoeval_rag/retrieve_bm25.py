#!/usr/bin/env python3
"""
BM25 retrieval for RepoEval using rank_bm25.BM25Okapi.

We retrieve within each repo:
  repo_id := task_id.rsplit('/', 1)[0]

Parallelism:
  - Multiprocessing across repos (each process builds BM25 once per repo and answers all tasks in that repo)

Inputs:
  - RepoEval parquet(s) for a given level
  - Repo-level segment cache built by build_library.py

Output:
  JSONL: one line per task
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import hashlib
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import pyarrow.dataset as ds
from rank_bm25 import BM25Okapi
from tqdm import tqdm


LANGUAGE = "python"
_TOKEN_RE = re.compile(r"[_A-Za-z][_A-Za-z0-9]*|\\d+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _repo_id(task_id: str) -> str:
    if "/" in task_id:
        return task_id.rsplit("/", 1)[0]
    return task_id


def _sanitize(name: str) -> str:
    safe = name.replace(os.sep, "__").replace("/", "__")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", safe)
    return safe


def _repo_filename(repo_id: str) -> str:
    prefix = _sanitize(repo_id)
    if len(prefix) > 80:
        prefix = prefix[:80]
    h = hashlib.sha1(repo_id.encode("utf-8", errors="replace")).hexdigest()[:10]
    return f"{prefix}__{h}.jsonl"


def _iter_parquet_rows(parquet_paths: List[str], columns: Optional[List[str]] = None, batch_size: int = 256) -> Iterable[Dict]:
    dataset = ds.dataset(parquet_paths, format="parquet")
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)
    for batch in scanner.to_batches():
        table = batch.to_pydict()
        n = len(next(iter(table.values()))) if table else 0
        for i in range(n):
            yield {k: table[k][i] for k in table.keys()}


def _load_repo_segments(repo_file: str) -> List[Dict]:
    segs: List[Dict] = []
    with open(repo_file, "r", encoding="utf-8") as f:
        for line in f:
            segs.append(json.loads(line))
    return segs


def _rank_all(scores) -> List[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


def _process_repo(args: Tuple[str, str, List[Tuple[int, str, str]], Optional[int]]) -> Tuple[str, List[Tuple[int, str, List[Dict]]]]:
    """
    Process one repo:
      args = (repo_id, repo_file, tasks[(idx, task_id, groundtruth)], topk)
    Returns:
      (repo_id, [(idx, task_id, results), ...])
    """
    repo_id, repo_file, tasks, topk = args
    segs = _load_repo_segments(repo_file)
    corpus_tokens = [_tokenize(s.get("text") or "") for s in segs]
    bm25 = BM25Okapi(corpus_tokens)

    out: List[Tuple[int, str, List[Dict]]] = []
    for idx, task_id, query in tasks:
        q_tokens = _tokenize(query)
        scores = bm25.get_scores(q_tokens)
        order = _rank_all(scores)
        if topk is not None:
            order = order[:topk]
        results: List[Dict] = []
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
        out.append((idx, task_id, results))
    return repo_id, out


def retrieve(
    parquet_paths: List[str],
    level: str,
    cache_dir: str,
    out_path: str,
    topk: int = 200,
    workers: int = 8,
    max_tasks: Optional[int] = None,
) -> None:
    level_dir = os.path.join(cache_dir, LANGUAGE, level)
    repos_dir = os.path.join(level_dir, "repos")
    if not os.path.isdir(repos_dir):
        raise FileNotFoundError(f"Repo cache not found: {repos_dir} (run build first)")

    cols = ["task_id", "groundtruth"]
    tasks: List[Tuple[int, str, str, str]] = []  # (idx, task_id, repo_id, query)
    for i, row in enumerate(_iter_parquet_rows(parquet_paths, columns=cols)):
        task_id = row["task_id"]
        query = row.get("groundtruth") or ""
        repo = _repo_id(task_id)
        tasks.append((i, task_id, repo, query))
        if max_tasks is not None and len(tasks) >= max_tasks:
            break

    by_repo: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
    for idx, task_id, repo, query in tasks:
        by_repo[repo].append((idx, task_id, query))

    # Build work items per repo
    work: List[Tuple[str, str, List[Tuple[int, str, str]], Optional[int]]] = []
    for repo, repo_tasks in by_repo.items():
        repo_file = os.path.join(repos_dir, _repo_filename(repo))
        if not os.path.exists(repo_file):
            raise FileNotFoundError(f"Missing repo cache file: {repo_file}")
        work.append((repo, repo_file, repo_tasks, topk))

    results_by_idx: List[Optional[Dict]] = [None] * len(tasks)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if workers == 1:
        for item in tqdm(work, desc=f"BM25 retrieve repos ({level})"):
            _repo, out_items = _process_repo(item)
            for idx, task_id, results in out_items:
                results_by_idx[idx] = {"task_id": task_id, "repo_id": _repo_id(task_id), "results": results}
    else:
        with mp.Pool(processes=workers) as pool:
            for _repo, out_items in tqdm(
                pool.imap_unordered(_process_repo, work, chunksize=1),
                total=len(work),
                desc=f"BM25 retrieve repos ({level}, workers={workers})",
            ):
                for idx, task_id, results in out_items:
                    results_by_idx[idx] = {"task_id": task_id, "repo_id": _repo_id(task_id), "results": results}

    with open(out_path, "w", encoding="utf-8") as out:
        for obj in results_by_idx:
            assert obj is not None
            out.write(json.dumps({"language": LANGUAGE, "level": level, **obj}, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, required=True, choices=["line_level", "api_level", "func_level"])
    parser.add_argument("--parquets", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="data/_repoeval_rag_cache")
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8, help="Processes (across repos)")
    parser.add_argument("--max_tasks", type=int, default=None)
    args = parser.parse_args()

    if args.parquets:
        parquet_paths = [p.strip() for p in args.parquets.split(",") if p.strip()]
    else:
        base = os.path.join("data", "repoeval", args.level)
        parquet_paths = sorted(os.path.join(base, f) for f in os.listdir(base) if f.endswith(".parquet"))

    out_path = args.out_path or os.path.join("data", f"_repoeval_rag_results_{args.level}.jsonl")
    retrieve(
        parquet_paths=parquet_paths,
        level=args.level,
        cache_dir=args.cache_dir,
        out_path=out_path,
        topk=args.topk,
        workers=args.workers,
        max_tasks=args.max_tasks,
    )
    print(f"Done. Wrote {out_path}")


if __name__ == "__main__":
    main()
