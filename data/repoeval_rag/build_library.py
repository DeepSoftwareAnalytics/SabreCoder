#!/usr/bin/env python3
"""
Build a repo-level retrieval library (RAG cache) for RepoEval.

RepoEval parquet schema matches CCEval:
  task_id, path, left_context, right_context, groundtruth, crossfile_context[{path,text}, ...]

We treat each repo as the retrieval unit:
  repo_id := task_id.rsplit('/', 1)[0]

For each repo, we parse each crossfile file ONCE with tree-sitter and extract semantic
segments (imports + function/method declarations). This dramatically reduces duplicated
work across many tasks from the same repo.

Cache layout:
  cache_dir/
    python/
      {level}/
        manifest.jsonl
        repos/
          {repo_id_sanitized}.jsonl    # one line per segment
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pyarrow.dataset as ds
from tqdm import tqdm
from tree_sitter_languages import get_parser


LANGUAGE = "python"


def _repo_id(task_id: str) -> str:
    if "/" in task_id:
        return task_id.rsplit("/", 1)[0]
    return task_id


def _sanitize(name: str) -> str:
    safe = name.replace(os.sep, "__").replace("/", "__")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", safe)
    return safe


def _repo_filename(repo_id: str) -> str:
    # Avoid collisions when different repo_ids map to the same sanitized name.
    # Keep a readable prefix + stable hash suffix.
    prefix = _sanitize(repo_id)
    if len(prefix) > 80:
        prefix = prefix[:80]
    h = hashlib.sha1(repo_id.encode("utf-8", errors="replace")).hexdigest()[:10]
    return f"{prefix}__{h}.jsonl"


def _iter_parquet_rows(parquet_paths: List[str], columns: Optional[List[str]] = None, batch_size: int = 128) -> Iterable[Dict]:
    dataset = ds.dataset(parquet_paths, format="parquet")
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)
    for batch in scanner.to_batches():
        table = batch.to_pydict()
        n = len(next(iter(table.values()))) if table else 0
        for i in range(n):
            yield {k: table[k][i] for k in table.keys()}


def _node_text(code: str, start_byte: int, end_byte: int) -> str:
    return code[start_byte:end_byte]


_NODE_TYPES = {
    "import": ["import_statement", "import_from_statement"],
    "function": ["function_definition"],
    "class": ["class_definition"],
}


def segment_file_parse_once(parser, text: str) -> List[Dict]:
    tree = parser.parse(text.encode("utf-8", errors="replace"))
    root = tree.root_node

    import_types = set(_NODE_TYPES["import"])
    function_types = set(_NODE_TYPES["function"])

    segments: List[Dict] = []
    stack = [root]
    while stack:
        node = stack.pop()
        t = node.type

        if t in import_types:
            seg_text = _node_text(text, node.start_byte, node.end_byte)
            if seg_text.strip():
                segments.append(
                    {
                        "start_line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                        "text": seg_text,
                        "segment_kind": "ts_node",
                        "node_type": t,
                    }
                )
            continue

        if t in function_types:
            seg_text = _node_text(text, node.start_byte, node.end_byte)
            if seg_text.strip():
                segments.append(
                    {
                        "start_line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                        "text": seg_text,
                        "segment_kind": "ts_node",
                        "node_type": t,
                    }
                )
            # Do not descend into function bodies
            continue

        for ch in node.children:
            stack.append(ch)

    segments.sort(key=lambda s: (s["start_line"], s["end_line"]))
    return segments


@dataclass(frozen=True)
class BuildStats:
    repos: int
    tasks: int
    files: int
    segments: int


def build_repo_library(
    parquet_paths: List[str],
    level: str,
    cache_dir: str,
    workers: int = 8,
    max_tasks: Optional[int] = None,
    max_crossfile_files_per_task: Optional[int] = None,
) -> BuildStats:
    if workers < 1:
        raise ValueError("--workers must be >= 1")

    level_dir = os.path.join(cache_dir, LANGUAGE, level)
    repos_dir = os.path.join(level_dir, "repos")
    os.makedirs(repos_dir, exist_ok=True)

    manifest_path = os.path.join(level_dir, "manifest.jsonl")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    # Thread-local parser
    _tls = threading.local()

    def _get_thread_parser():
        p = getattr(_tls, "parser", None)
        if p is None:
            _tls.parser = get_parser(LANGUAGE)
            p = _tls.parser
        return p

    def _segment_one_file(cf: Dict) -> Tuple[str, List[Dict]]:
        file_path = cf.get("path") or ""
        text = cf.get("text") or ""
        segs = segment_file_parse_once(_get_thread_parser(), text)
        return file_path, segs

    # Repo state: seen file paths + counters
    repo_seen_files: Dict[str, Set[str]] = {}
    repo_seg_counts: Dict[str, int] = {}
    repo_file_counts: Dict[str, int] = {}

    tasks_seen = 0
    files_total = 0
    segments_total = 0

    cols = ["task_id", "crossfile_context"]
    for row in tqdm(_iter_parquet_rows(parquet_paths, columns=cols), desc=f"Build RepoEval RAG cache ({level})"):
        tasks_seen += 1
        if max_tasks is not None and tasks_seen > max_tasks:
            break

        task_id = row["task_id"]
        repo = _repo_id(task_id)

        if repo not in repo_seen_files:
            repo_seen_files[repo] = set()
            repo_seg_counts[repo] = 0
            repo_file_counts[repo] = 0

        crossfiles = row.get("crossfile_context") or []
        if max_crossfile_files_per_task is not None:
            crossfiles = crossfiles[: max_crossfile_files_per_task]

        # Filter out files already seen for this repo
        new_files = [cf for cf in crossfiles if (cf.get("path") or "") not in repo_seen_files[repo]]
        if not new_files:
            continue

        repo_file = os.path.join(repos_dir, _repo_filename(repo))
        os.makedirs(os.path.dirname(repo_file), exist_ok=True)

        # Segment in parallel (threaded)
        if workers == 1:
            segmented = [_segment_one_file(cf) for cf in new_files]
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                segmented = list(ex.map(_segment_one_file, new_files))

        # Append deterministically by input order
        with open(repo_file, "a", encoding="utf-8") as out:
            for file_path, segs in segmented:
                if file_path in repo_seen_files[repo]:
                    continue
                repo_seen_files[repo].add(file_path)
                repo_file_counts[repo] += 1
                files_total += 1

                for s in segs:
                    out.write(
                        json.dumps(
                            {
                                "repo_id": repo,
                                "language": LANGUAGE,
                                "level": level,
                                "file_path": file_path,
                                "start_line": s["start_line"],
                                "end_line": s["end_line"],
                                "text": s["text"],
                                "segment_kind": s["segment_kind"],
                                "node_type": s["node_type"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    repo_seg_counts[repo] += 1
                    segments_total += 1

    # Write manifest
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for repo in sorted(repo_seen_files.keys()):
            mf.write(
                json.dumps(
                    {
                        "repo_id": repo,
                        "language": LANGUAGE,
                        "level": level,
                        "files": repo_file_counts[repo],
                        "segments": repo_seg_counts[repo],
                        "repo_file": os.path.relpath(os.path.join(repos_dir, _repo_filename(repo)), level_dir),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return BuildStats(repos=len(repo_seen_files), tasks=tasks_seen, files=files_total, segments=segments_total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, required=True, choices=["line_level", "api_level", "func_level"])
    parser.add_argument("--parquets", type=str, default=None, help="Comma-separated parquet paths; default uses data/repoeval/{level}/*.parquet")
    parser.add_argument("--cache_dir", type=str, default="data/_repoeval_rag_cache")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--max_crossfile_files_per_task", type=int, default=None)
    args = parser.parse_args()

    if args.parquets:
        parquet_paths = [p.strip() for p in args.parquets.split(",") if p.strip()]
    else:
        base = os.path.join("data", "repoeval", args.level)
        parquet_paths = sorted(
            os.path.join(base, f) for f in os.listdir(base) if f.endswith(".parquet")
        )

    stats = build_repo_library(
        parquet_paths=parquet_paths,
        level=args.level,
        cache_dir=args.cache_dir,
        workers=args.workers,
        max_tasks=args.max_tasks,
        max_crossfile_files_per_task=args.max_crossfile_files_per_task,
    )
    print(
        f"Done. level={args.level} repos={stats.repos} tasks={stats.tasks} files={stats.files} segments={stats.segments} "
        f"cache_dir={args.cache_dir}/{LANGUAGE}/{args.level}"
    )


if __name__ == "__main__":
    main()
