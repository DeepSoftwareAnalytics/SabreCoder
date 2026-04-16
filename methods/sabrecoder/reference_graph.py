"""
Tree-sitter Cross-Reference Analyzer (Multi-Language)

This module extracts cross-references between code chunks using tree-sitter.
It supports Python / Java / C# for:
  - Function/method calls (best-effort)
  - Class inheritance (Java/C# best-effort)

Design goals:
  - No silent try/except fallbacks: parsing and traversal are deterministic.
  - Best-effort extraction even when the parse tree contains errors (common for prefix-code prompts).
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from tree_sitter_languages import get_parser


@dataclass
class CrossReference:
    """Represents a cross-reference relationship between chunks"""

    source_chunk_idx: int  # Chunk that contains the reference
    target_chunk_idx: int  # Chunk being referenced
    ref_type: str  # Type: 'call', 'inherit'
    name: str  # Name of the referenced entity
    line_number: int  # Line where reference occurs (1-based)
    confidence: float = 1.0  # Confidence score (0-1)

    def __repr__(self):
        return f"<{self.ref_type}: chunk[{self.source_chunk_idx}] -> chunk[{self.target_chunk_idx}] ({self.name})>"


def _normalize_language(language: Optional[str]) -> str:
    if not language:
        return ""
    language = language.lower()
    if language in {"csharp", "c#", "c_sharp"}:
        return "c_sharp"
    if language in {"python", "java"}:
        return language
    return language


def _detect_language_from_code(code: str) -> str:
    # Keep consistent with TreeSitterSegmenter.detect_language, but avoid importing it here.
    markers = {
        "c_sharp": ["using System", "namespace ", "public class", "private class"],
        "java": ["package ", "import java.", "public class", "private class", "public static void"],
        "python": ["def ", "class ", "import ", "from "],
    }
    for lang in ["c_sharp", "java", "python"]:
        for m in markers.get(lang, []):
            if m in code:
                return lang
    return "python"


def _node_text(code_bytes: bytes, start_byte: int, end_byte: int) -> str:
    # tree-sitter byte offsets are in UTF-8 bytes; do not slice Python str by bytes.
    return code_bytes[start_byte:end_byte].decode("utf-8", errors="replace")


def _last_identifier_in_subtree(node, code_bytes: bytes) -> Optional[str]:
    """
    Return the last identifier-ish token in the subtree.
    This is a pragmatic way to get callee/type names across languages.
    """
    best: Optional[Tuple[int, str]] = None  # (end_byte, text)
    stack = [node]
    while stack:
        n = stack.pop()
        t = n.type
        if t in {"identifier", "type_identifier"}:
            text = _node_text(code_bytes, n.start_byte, n.end_byte)
            if best is None or n.end_byte > best[0]:
                best = (n.end_byte, text)
        for ch in n.children:
            stack.append(ch)
    return best[1] if best else None


def _extract_call_name(node, code_bytes: bytes, language: str) -> Optional[str]:
    """
    Extract best-effort callee name for a call-like node.
    """
    if language == "python":
        # tree-sitter-python: call(function: (expression), arguments: (argument_list))
        fn = node.child_by_field_name("function")
        if fn is None:
            return _last_identifier_in_subtree(node, code_bytes)
        return _last_identifier_in_subtree(fn, code_bytes)

    if language == "java":
        # tree-sitter-java: method_invocation(name: (identifier)), object_creation_expression(type: ...)
        name = node.child_by_field_name("name")
        if name is not None:
            return _node_text(code_bytes, name.start_byte, name.end_byte)
        expr = node.child_by_field_name("object")
        if expr is not None:
            return _last_identifier_in_subtree(expr, code_bytes)
        return _last_identifier_in_subtree(node, code_bytes)

    if language == "c_sharp":
        # tree-sitter-c-sharp: invocation_expression(expression: ...), object_creation_expression(type: ...)
        expr = node.child_by_field_name("expression")
        if expr is not None:
            return _last_identifier_in_subtree(expr, code_bytes)
        return _last_identifier_in_subtree(node, code_bytes)

    return _last_identifier_in_subtree(node, code_bytes)


def _extract_inherit_names(node, code_bytes: bytes, language: str) -> List[str]:
    """
    Extract best-effort base type names from a class declaration node.
    """
    names: List[str] = []
    if language == "java":
        # tree-sitter-java: class_declaration(superclass: (superclass (type_identifier)))
        superclass = node.child_by_field_name("superclass")
        if superclass is not None:
            name = _last_identifier_in_subtree(superclass, code_bytes)
            if name:
                names.append(name)
        interfaces = node.child_by_field_name("interfaces")
        if interfaces is not None:
            # interface list contains type identifiers
            stack = [interfaces]
            while stack:
                n = stack.pop()
                if n.type in {"type_identifier"}:
                    names.append(_node_text(code_bytes, n.start_byte, n.end_byte))
                for ch in n.children:
                    stack.append(ch)
        return names

    if language == "c_sharp":
        # tree-sitter-c-sharp: class_declaration(base_list: (base_list ...))
        base_list = node.child_by_field_name("base_list")
        if base_list is None:
            # Some grammars use 'bases' or unnamed; fall back to subtree scan.
            base_list = next((c for c in node.children if c.type == "base_list"), None)
        if base_list is not None:
            stack = [base_list]
            while stack:
                n = stack.pop()
                if n.type in {"identifier", "type_identifier"}:
                    names.append(_node_text(code_bytes, n.start_byte, n.end_byte))
                for ch in n.children:
                    stack.append(ch)
        return names

    return names


class CrossReferenceAnalyzer:
    """
    Analyze cross-references between chunks using tree-sitter.

    Notes:
      - This is best-effort and name-based; it does not resolve namespaces/types.
      - For prefix-code prompts, the parse tree may contain errors; we still try to extract references.
    """

    def __init__(self, chunks, code: str, language: Optional[str] = None):
        self.chunks = chunks
        self.code = code
        self.code_bytes = code.encode("utf-8", errors="replace")
        self.language = _normalize_language(language) or _detect_language_from_code(code)

        self.source_lines = code.split("\n")

        self.chunk_by_name: Dict[str, List[int]] = defaultdict(list)  # name -> [chunk_indices]
        self.chunk_by_line: Dict[int, int] = {}  # line_number -> chunk_index (1-based)

        self._build_indices()

        self.references: List[CrossReference] = []

    def _build_indices(self) -> None:
        for idx, chunk in enumerate(self.chunks):
            name = getattr(chunk, "name", "") or ""
            if name:
                self.chunk_by_name[name].append(idx)

            start_line = int(getattr(chunk, "start_line", 0) or 0)
            end_line = int(getattr(chunk, "end_line", 0) or 0)
            for line in range(start_line, end_line + 1):
                self.chunk_by_line[line] = idx

    def analyze(self) -> List[CrossReference]:
        parser = get_parser(self.language)
        tree = parser.parse(self.code_bytes)
        root = tree.root_node

        self.references = []

        call_node_types = {
            "python": {"call"},
            "java": {"method_invocation", "object_creation_expression", "explicit_constructor_invocation"},
            "c_sharp": {"invocation_expression", "object_creation_expression"},
        }.get(self.language, set())

        class_node_types = {
            "python": {"class_definition"},
            "java": {"class_declaration", "interface_declaration"},
            "c_sharp": {"class_declaration", "interface_declaration", "struct_declaration"},
        }.get(self.language, set())

        stack = [root]
        while stack:
            node = stack.pop()
            node_type = node.type

            # Calls / object creation
            if node_type in call_node_types:
                line = node.start_point[0] + 1
                source_idx = self.chunk_by_line.get(line)
                if source_idx is not None:
                    name = _extract_call_name(node, self.code_bytes, self.language)
                    if name and name in self.chunk_by_name:
                        for target_idx in self.chunk_by_name[name]:
                            if target_idx == source_idx:
                                continue
                            self.references.append(
                                CrossReference(
                                    source_chunk_idx=source_idx,
                                    target_chunk_idx=target_idx,
                                    ref_type="call",
                                    name=name,
                                    line_number=line,
                                    confidence=1.0,
                                )
                            )

            # Inheritance (Java/C# best-effort)
            if node_type in class_node_types:
                line = node.start_point[0] + 1
                source_idx = self.chunk_by_line.get(line)
                if source_idx is not None:
                    for base_name in _extract_inherit_names(node, self.code_bytes, self.language):
                        if base_name in self.chunk_by_name:
                            for target_idx in self.chunk_by_name[base_name]:
                                if target_idx == source_idx:
                                    continue
                                self.references.append(
                                    CrossReference(
                                        source_chunk_idx=source_idx,
                                        target_chunk_idx=target_idx,
                                        ref_type="inherit",
                                        name=base_name,
                                        line_number=line,
                                        confidence=1.0,
                                    )
                                )

            # Traversal
            for ch in node.children:
                stack.append(ch)

        return self.references

    def get_call_graph(self) -> Dict[int, Set[int]]:
        graph: Dict[int, Set[int]] = defaultdict(set)
        for ref in self.references:
            if ref.ref_type == "call" and ref.confidence >= 0.5:
                graph[ref.source_chunk_idx].add(ref.target_chunk_idx)
        return dict(graph)

    def get_reverse_call_graph(self) -> Dict[int, Set[int]]:
        graph: Dict[int, Set[int]] = defaultdict(set)
        for ref in self.references:
            if ref.ref_type == "call" and ref.confidence >= 0.5:
                graph[ref.target_chunk_idx].add(ref.source_chunk_idx)
        return dict(graph)
