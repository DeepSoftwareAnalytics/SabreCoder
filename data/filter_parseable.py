#!/usr/bin/env python3
"""
Filter dataset samples that are parseable by tree-sitter for SabreCoder.

Targets:
  - LCC_python / LCC_java / LCC_csharp: only validation + test splits (have GT)
  - CCEval (repo-level completion): data/cceval/{java,python}/test.parquet

Definition of "parseable" (step-1):
  - Use tree_sitter_languages.get_parser(lang).parse(code_bytes)
  - Accept a sample if:
      * it has no hard parse errors (only missing nodes near EOF), OR
      * we can still extract at least one semantic node (import/class/function).
  - If direct parsing is problematic, retry by truncating the last 1~3 lines and parsing again.
    * The removed tail lines are preserved in the output for later chunking/analysis.

Outputs:
  - Writes JSONL files compatible with SabreCoder evaluation
    (expects: context + ground_truth; for cceval also provides crossfile_context/current_file_context).

Example:
  python data/filter_parseable.py --out_dir data/_filtered_ts
  python data/filter_parseable.py --out_dir data/_filtered_ts --limit 200
  python data/filter_parseable.py --out_dir data/_filtered_ts --max_crossfile_items 8
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tree_sitter_languages import get_parser


@dataclass(frozen=True)
class ParseResult:
    ok: bool
    truncated_lines: int  # 0..3
    parse_code: str  # code that parses cleanly (maybe truncated)
    tail_code: str  # removed tail (maybe empty)
    issues_ok: bool  # no hard errors; missing nodes only near EOF
    chunk_count: int  # number of semantic nodes (import/class/function) found


def _normalize_language(language: str) -> str:
    language = (language or "").lower()
    if language in {"csharp", "c#", "c_sharp"}:
        return "c_sharp"
    if language in {"python", "java"}:
        return language
    raise ValueError(f"Unsupported language: {language}")


def _split_lines(code: str) -> List[str]:
    # Keep behavior similar to existing AST-based logic in repo (split on '\n').
    return code.split("\n")


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


def _count_semantic_nodes(root, language: str) -> int:
    node_types = _NODE_TYPES.get(language, {})
    import_types = set(node_types.get("import", []))
    class_types = set(node_types.get("class", []))
    func_types = set(node_types.get("function", []))

    count = 0
    stack = [root]
    while stack:
        n = stack.pop()
        t = n.type
        if t in import_types or t in class_types or t in func_types:
            count += 1
            # Do not descend into function bodies when counting top-level semantic nodes.
            if t in func_types:
                continue
        for ch in n.children:
            stack.append(ch)
    return count


def try_tree_sitter_parse(code: str, language: str, max_truncate_lines: int = 3) -> ParseResult:
    language = _normalize_language(language)
    parser = get_parser(language)

    lines = _split_lines(code)
    best: Optional[ParseResult] = None

    for remove_count in [0, 1, 2, 3]:
        if remove_count > max_truncate_lines:
            break
        if remove_count >= len(lines):
            break

        if remove_count == 0:
            candidate = code
            tail = ""
        else:
            candidate = "\n".join(lines[:-remove_count])
            tail = "\n".join(lines[-remove_count:])

        tree = parser.parse(candidate.encode("utf-8", errors="replace"))
        root = tree.root_node

        # Tree-sitter often reports has_error=True for incomplete *prefix* files (e.g., missing trailing '}').
        # Missing nodes near EOF are acceptable; hard ERROR nodes are not.
        num_lines = len(_split_lines(candidate))
        allowed_missing_tail_lines = 3

        issues_ok = True
        stack = [root]
        while stack:
            n = stack.pop()
            if getattr(n, "is_error", False) or n.type == "ERROR":
                issues_ok = False
                break
            if getattr(n, "is_missing", False):
                missing_line = n.start_point[0] + 1
                if missing_line < max(1, num_lines - allowed_missing_tail_lines + 1):
                    issues_ok = False
                    break
            for ch in n.children:
                stack.append(ch)

        chunk_count = _count_semantic_nodes(root, language=language)
        keep = issues_ok or (chunk_count > 0)
        if not keep:
            continue

        pr = ParseResult(
            ok=True,
            truncated_lines=remove_count,
            parse_code=candidate,
            tail_code=tail,
            issues_ok=issues_ok,
            chunk_count=chunk_count,
        )

        if best is None:
            best = pr
            if remove_count == 0 and issues_ok:
                return best
            continue

        # Prefer candidates that "compile" cleanly (issues_ok), then lower truncation, then more chunks.
        pr_key = (0 if pr.issues_ok else 1, pr.truncated_lines, -pr.chunk_count)
        best_key = (0 if best.issues_ok else 1, best.truncated_lines, -best.chunk_count)
        if pr_key < best_key:
            best = pr

    if best is not None:
        return best

    return ParseResult(
        ok=False,
        truncated_lines=-1,
        parse_code="",
        tail_code="",
        issues_ok=False,
        chunk_count=0,
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_jsonl(records: Iterable[Dict[str, Any]], out_path: str) -> Tuple[int, int]:
    kept = 0
    total = 0
    _ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            total += 1
            if rec.get("_keep", False):
                kept += 1
                rec = dict(rec)
                rec.pop("_keep", None)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return kept, total


def _format_crossfile_context(items: List[Dict[str, Any]], language: str) -> str:
    """
    Convert CCEval crossfile_context fragments into a prompt-friendly (commented) string.

    Notes:
      - This text is not parsed. SabreCoder only parses `current_file_context` when crossfile fields are provided.
      - We comment the crossfile code to avoid breaking the prompt's syntax.
    """
    language = (language or "").lower()
    comment = "# " if language == "python" else "// "

    if not items:
        return ""

    parts: List[str] = []
    parts.append(comment + "Here are some relevant code fragments from other files of the repo:\n\n")

    for it in items:
        path = (it.get("path") or "").strip()
        text = it.get("text") or ""
        parts.append(comment + "the below code fragment can be found in:\n")
        if path:
            parts.append(comment + f"{path}\n")
        for line in text.split("\n"):
            parts.append(comment + line + "\n")
        parts.append("\n")

    parts.append("\n")
    return "".join(parts)


def _iter_lcc_parquet_rows(parquet_path: str) -> Iterable[Dict[str, Any]]:
    table = pq.ParquetFile(parquet_path)
    for batch in table.iter_batches(batch_size=256):
        cols = batch.to_pydict()
        num = len(next(iter(cols.values()))) if cols else 0
        for i in range(num):
            yield {k: cols[k][i] for k in cols.keys()}


def filter_lcc_split(
    dataset_dir: str,
    split: str,
    language: str,
    out_path: str,
    limit: Optional[int] = None,
) -> Tuple[int, int]:
    data_dir = os.path.join(dataset_dir, "data")
    files = sorted(
        f for f in os.listdir(data_dir) if f.startswith(split + "-") and f.endswith(".parquet")
    )
    parquet_paths = [os.path.join(data_dir, f) for f in files]

    def records() -> Iterable[Dict[str, Any]]:
        idx = 0
        for p in parquet_paths:
            for row in _iter_lcc_parquet_rows(p):
                if limit is not None and idx >= limit:
                    return
                idx += 1

                context = row.get("context", "")
                gt = row.get("gt", "")

                pr = try_tree_sitter_parse(context, language)
                keep = pr.ok

                yield {
                    "_keep": keep,
                    "context": context,
                    "ground_truth": gt,
                    "metadata": {
                        "dataset": os.path.basename(dataset_dir),
                        "split": split,
                        "row_idx": idx - 1,
                        "language": language,
                        "tree_sitter_truncated_lines": pr.truncated_lines if pr.ok else None,
                    },
                    # Keep for later “tail as last chunk” handling (if needed)
                    "structure_parse_code": pr.parse_code if pr.ok else None,
                    "structure_incomplete_tail": pr.tail_code if pr.ok else None,
                }

    kept, total = _write_jsonl(records(), out_path)
    return kept, total


def filter_cceval(
    parquet_path: str,
    language: str,
    out_path: str,
    limit: Optional[int] = None,
    max_crossfile_items: Optional[int] = None,
) -> Tuple[int, int]:
    language_norm = _normalize_language(language)

    dataset = ds.dataset(parquet_path, format="parquet")

    def records() -> Iterable[Dict[str, Any]]:
        idx = 0
        scanner = dataset.scanner(batch_size=128)
        for batch in scanner.to_batches():
            table = batch.to_pydict()
            num = len(next(iter(table.values()))) if table else 0
            for i in range(num):
                if limit is not None and idx >= limit:
                    return
                idx += 1

                task_id = table.get("task_id", [None])[i]
                path = table.get("path", [None])[i]
                left_context = table.get("left_context", [""])[i] or ""
                right_context = table.get("right_context", [""])[i] or ""
                groundtruth = table.get("groundtruth", [""])[i] or ""
                crossfile_items = table.get("crossfile_context", [[]])[i] or []

                # Optionally cap crossfile items to reduce prompt size (if desired)
                if max_crossfile_items is not None:
                    crossfile_items = crossfile_items[: max_crossfile_items]

                # Parse only the current file prefix (left context)
                pr = try_tree_sitter_parse(left_context, language_norm)
                keep = pr.ok

                crossfile_text = _format_crossfile_context(crossfile_items, language=language)
                current_code = left_context
                full_prompt = crossfile_text + current_code

                yield {
                    "_keep": keep,
                    "context": full_prompt,
                    "ground_truth": groundtruth,
                    "right_context": right_context,
                    # The mask builder uses these fields to restrict sparse rules to the current file.
                    "crossfile_context": crossfile_text,
                    "current_file_context": current_code,
                    "metadata": {
                        "dataset": "cceval",
                        "language": language,
                        "row_idx": idx - 1,
                        "task_id": task_id,
                        "path": path,
                        "tree_sitter_truncated_lines": pr.truncated_lines if pr.ok else None,
                        "crossfile_items": len(crossfile_items),
                    },
                    # Keep for later “tail as last chunk” handling (if needed)
                    "structure_parse_code": pr.parse_code if pr.ok else None,
                    "structure_incomplete_tail": pr.tail_code if pr.ok else None,
                }

    kept, total = _write_jsonl(records(), out_path)
    return kept, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("data", "_filtered_ts"),
        help="Output directory for filtered JSONL files",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on processed rows per split")
    parser.add_argument(
        "--max_crossfile_items",
        type=int,
        default=None,
        help="Optional cap on crossfile fragments per sample (CCEval only)",
    )
    parser.add_argument("--no_lcc", action="store_true", help="Skip LCC datasets")
    parser.add_argument("--no_cceval", action="store_true", help="Skip CCEval datasets")
    args = parser.parse_args()

    out_dir = args.out_dir
    _ensure_dir(out_dir)

    summary: List[Tuple[str, int, int, str]] = []

    if not args.no_lcc:
        lcc_jobs = [
            ("data/LCC_python", "python"),
            ("data/LCC_java", "java"),
            ("data/LCC_csharp", "c_sharp"),
        ]
        for ds_dir, lang in lcc_jobs:
            for split in ["validation", "test"]:
                out_path = os.path.join(
                    out_dir,
                    f"{os.path.basename(ds_dir)}_{split}_parseable.jsonl",
                )
                kept, total = filter_lcc_split(
                    dataset_dir=ds_dir,
                    split=split,
                    language=lang,
                    out_path=out_path,
                    limit=args.limit,
                )
                summary.append((f"{ds_dir}:{split}", kept, total, out_path))

    if not args.no_cceval:
        cceval_jobs = [
            ("data/cceval/python/test.parquet", "python"),
            ("data/cceval/java/test.parquet", "java"),
        ]
        for parquet_path, lang in cceval_jobs:
            out_path = os.path.join(out_dir, f"cceval_{lang}_test_parseable.jsonl")
            kept, total = filter_cceval(
                parquet_path=parquet_path,
                language=lang,
                out_path=out_path,
                limit=args.limit,
                max_crossfile_items=args.max_crossfile_items,
            )
            summary.append((f"{parquet_path}", kept, total, out_path))

    print("\nDone. Filter summary:")
    for name, kept, total, out_path in summary:
        ratio = (kept / total * 100.0) if total else 0.0
        print(f"  - {name}: kept {kept}/{total} ({ratio:.2f}%) -> {out_path}")


if __name__ == "__main__":
    main()
