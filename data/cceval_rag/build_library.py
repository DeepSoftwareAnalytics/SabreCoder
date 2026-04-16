#!/usr/bin/env python3
"""
Build a per-task retrieval library (RAG cache) from CCEval parquet.

Default segmentation mode: parse each crossfile file ONCE with tree-sitter and
extract semantic segments from the syntax tree.

Segmentation modes:
  - parse_once (default): parse full file once, extract import/function/method nodes.
  - incremental: your original rule (grow [i..j] until parseable, else force max_lines).

Outputs (cache):
  out_dir/
    {language}/
      manifest.jsonl
      tasks/
        {task_id_sanitized}.jsonl     # one line per segment

Each segment line contains:
  - task_id, language, file_path, start_line, end_line, text, segment_kind, node_type
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pyarrow.dataset as ds
from tqdm import tqdm
from tree_sitter_languages import get_parser


def _normalize_language(language: str) -> str:
    language = (language or "").lower()
    if language in {"python", "java"}:
        return language
    if language in {"csharp", "c#", "c_sharp"}:
        return "c_sharp"
    raise ValueError(f"Unsupported language: {language}")


def _task_filename(task_id: str) -> str:
    safe = task_id.replace(os.sep, "__").replace("/", "__")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", safe)
    return safe + ".jsonl"


def _iter_parquet_rows(parquet_path: str, columns: Optional[List[str]] = None, batch_size: int = 128) -> Iterable[Dict]:
    dataset = ds.dataset(parquet_path, format="parquet")
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)
    for batch in scanner.to_batches():
        table = batch.to_pydict()
        n = len(next(iter(table.values()))) if table else 0
        for i in range(n):
            yield {k: table[k][i] for k in table.keys()}


def _node_text(code: str, start_byte: int, end_byte: int) -> str:
    return code[start_byte:end_byte]


def _is_parseable(parser, snippet: str) -> bool:
    tree = parser.parse(snippet.encode("utf-8", errors="replace"))
    return not tree.root_node.has_error


_NODE_TYPES: Dict[str, Dict[str, List[str]]] = {
    "python": {
        "function": ["function_definition"],
        "class": ["class_definition"],
        "import": ["import_statement", "import_from_statement"],
    },
    "java": {
        "function": ["method_declaration", "constructor_declaration"],
        "class": ["class_declaration", "interface_declaration", "enum_declaration"],
        "import": ["import_declaration", "package_declaration"],
    },
    "c_sharp": {
        "function": ["method_declaration", "constructor_declaration"],
        "class": ["class_declaration", "interface_declaration", "struct_declaration"],
        "import": ["using_directive"],
    },
}


def segment_file_parse_once(parser, language: str, text: str) -> List[Dict]:
    """
    Parse full text once, extract semantic nodes into segments.
    Excludes whitespace-only segments.
    """
    tree = parser.parse(text.encode("utf-8", errors="replace"))
    root = tree.root_node

    node_types = _NODE_TYPES.get(language, {})
    import_types = set(node_types.get("import", []))
    function_types = set(node_types.get("function", []))
    class_types = set(node_types.get("class", []))

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
            # Do not descend into bodies
            continue

        # We do NOT add class nodes by default (avoid overlaps); we still descend to find methods.
        # If you want class-level segments, add a flag and include t in class_types here.

        for ch in node.children:
            stack.append(ch)

    # Sort by (start_line,end_line) for deterministic order within a file.
    segments.sort(key=lambda s: (s["start_line"], s["end_line"]))
    return segments


def segment_file_incremental_parse(parser, text: str, max_lines: int) -> List[Dict]:
    """
    Your original incremental parse rule (line-growing).
    Excludes empty / whitespace-only lines from becoming standalone segments.
    """
    lines = text.split("\n")
    n = len(lines)
    segments: List[Dict] = []

    i = 1
    while i <= n:
        # Skip empty / whitespace-only lines entirely.
        while i <= n and lines[i - 1].strip() == "":
            i += 1
        if i > n:
            break

        end_limit = min(n, i + max_lines - 1)
        chosen_end: Optional[int] = None
        chosen_text: Optional[str] = None

        for j in range(i, end_limit + 1):
            snippet = "\n".join(lines[i - 1 : j])
            if snippet.strip() == "":
                continue
            if _is_parseable(parser, snippet):
                chosen_end = j
                chosen_text = snippet
                break

        if chosen_end is None:
            chosen_end = end_limit
            chosen_text = "\n".join(lines[i - 1 : chosen_end])

        if chosen_text.strip():
            segments.append(
                {
                    "start_line": i,
                    "end_line": chosen_end,
                    "text": chosen_text,
                    "segment_kind": "incremental",
                    "node_type": None,
                }
            )

        i = chosen_end + 1

    return segments


@dataclass(frozen=True)
class BuildStats:
    tasks: int
    segments: int
    files: int


def build_library(
    parquet_path: str,
    language: str,
    out_dir: str,
    mode: str = "parse_once",
    max_lines: int = 10,
    workers: int = 1,
    max_tasks: Optional[int] = None,
    max_crossfile_files_per_task: Optional[int] = None,
) -> BuildStats:
    language = _normalize_language(language)
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    if mode not in {"parse_once", "incremental"}:
        raise ValueError("--mode must be parse_once or incremental")

    # Thread-local parser to avoid shared mutable parser state across threads.
    _tls = threading.local()

    def _get_thread_parser():
        p = getattr(_tls, "parser", None)
        if p is None:
            _tls.parser = get_parser(language)
            p = _tls.parser
        return p

    def _segment_one_file(idx_and_cf: Tuple[int, Dict]) -> Tuple[int, str, List[Dict]]:
        idx, cf = idx_and_cf
        file_path = cf.get("path") or ""
        text = cf.get("text") or ""
        parser_local = _get_thread_parser()
        if mode == "parse_once":
            segs = segment_file_parse_once(parser_local, language=language, text=text)
        else:
            segs = segment_file_incremental_parse(parser_local, text=text, max_lines=max_lines)
        return idx, file_path, segs

    lang_dir = os.path.join(out_dir, language)
    tasks_dir = os.path.join(lang_dir, "tasks")
    os.makedirs(tasks_dir, exist_ok=True)

    manifest_path = os.path.join(lang_dir, "manifest.jsonl")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    tasks_written = 0
    segments_written = 0
    files_seen = 0

    needed_cols = ["task_id", "crossfile_context"]
    row_iter = _iter_parquet_rows(parquet_path, columns=needed_cols)

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for row in tqdm(row_iter, desc=f"Build RAG library ({language}, mode={mode})"):
            task_id = row["task_id"]
            crossfiles = row.get("crossfile_context") or []

            if max_crossfile_files_per_task is not None:
                crossfiles = crossfiles[: max_crossfile_files_per_task]

            task_file = os.path.join(tasks_dir, _task_filename(task_id))
            if os.path.exists(task_file):
                os.remove(task_file)

            seg_count_this_task = 0
            file_count_this_task = 0

            indexed = list(enumerate(crossfiles))
            if workers == 1:
                results = [_segment_one_file(x) for x in indexed]
            else:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    results = list(ex.map(_segment_one_file, indexed))

            results.sort(key=lambda x: x[0])  # deterministic by crossfile order

            with open(task_file, "w", encoding="utf-8") as tf:
                for _idx, file_path, segs in results:
                    file_count_this_task += 1
                    files_seen += 1
                    for s in segs:
                        tf.write(
                            json.dumps(
                                {
                                    "task_id": task_id,
                                    "language": language,
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
                        seg_count_this_task += 1
                        segments_written += 1

            mf.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "language": language,
                        "segments": seg_count_this_task,
                        "files": file_count_this_task,
                        "task_file": os.path.relpath(task_file, lang_dir),
                        "mode": mode,
                        "max_lines": max_lines,
                        "workers": workers,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            tasks_written += 1
            if max_tasks is not None and tasks_written >= max_tasks:
                break

    return BuildStats(tasks=tasks_written, segments=segments_written, files=files_seen)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, required=True, help="Path to CCEval test.parquet")
    parser.add_argument("--language", type=str, required=True, choices=["python", "java"], help="Language")
    parser.add_argument("--out_dir", type=str, default="data/_cceval_rag_cache", help="Output cache directory")
    parser.add_argument("--mode", type=str, default="parse_once", choices=["parse_once", "incremental"])
    parser.add_argument("--max_lines", type=int, default=10, help="Max lines before forcing a segment (incremental)")
    parser.add_argument("--workers", type=int, default=1, help="Thread workers for per-task crossfile segmentation")
    parser.add_argument("--max_tasks", type=int, default=None, help="Optional cap for debugging")
    parser.add_argument("--max_crossfile_files_per_task", type=int, default=None, help="Optional cap for debugging")
    args = parser.parse_args()

    stats = build_library(
        parquet_path=args.parquet,
        language=args.language,
        out_dir=args.out_dir,
        mode=args.mode,
        max_lines=args.max_lines,
        workers=args.workers,
        max_tasks=args.max_tasks,
        max_crossfile_files_per_task=args.max_crossfile_files_per_task,
    )
    print(
        f"Done. tasks={stats.tasks} files={stats.files} segments={stats.segments} "
        f"cache_dir={args.out_dir}/{args.language}"
    )


if __name__ == "__main__":
    main()

