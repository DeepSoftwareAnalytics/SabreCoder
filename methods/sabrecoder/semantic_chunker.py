"""
Tree-sitter Based Multi-Language Code Segmentation

This module uses tree-sitter to parse code from multiple programming languages
and segments it into meaningful chunks based on:
- Function definitions
- Class definitions
- Methods within classes
- Import statements

Supported languages: Python, Java, C#
"""

from tree_sitter_languages import get_parser
from tree_sitter import Node
from typing import List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass


class ChunkType(Enum):
    """Types of code chunks"""
    IMPORT = "import"
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE_CODE = "module_code"
    COMMENT = "comment"
    DOCSTRING = "docstring"


@dataclass
class CodeChunk:
    """Represents a semantically meaningful span of code."""
    chunk_type: ChunkType
    start: int  # byte offset
    end: int  # byte offset
    text: str  # The actual code content
    name: str = ""  # Function/class name
    parent_name: Optional[str] = None  # For methods, the parent class name
    start_line: int = 0
    end_line: int = 0
    # Definition/signature line range for functions, classes, and methods.
    signature_start_line: Optional[int] = None
    signature_end_line: Optional[int] = None


class TreeSitterSegmenter:
    """Multi-language code segmenter using tree-sitter"""

    # Language detection markers
    LANGUAGE_MARKERS = {
        'java': ['package ', 'import java.', 'public class', 'private class', 'public static void'],
        'c_sharp': ['using System', 'namespace ', 'public class', 'private class'],
        'python': ['def ', 'class ', 'import ', 'from ']
    }

    # Node types for different constructs in each language
    NODE_TYPES = {
        'python': {
            'function': ['function_definition'],
            'class': ['class_definition'],
            'import': ['import_statement', 'import_from_statement'],
        },
        'java': {
            'function': ['method_declaration', 'constructor_declaration'],
            'class': ['class_declaration', 'interface_declaration', 'enum_declaration'],
            'import': ['import_declaration', 'package_declaration'],
        },
        'c_sharp': {
            'function': ['method_declaration', 'constructor_declaration'],
            'class': ['class_declaration', 'interface_declaration', 'struct_declaration'],
            'import': ['using_directive'],
        }
    }

    def __init__(self, tokenizer=None):
        """
        Initialize the tree-sitter segmenter

        Args:
            tokenizer: Optional tokenizer for token counting (not used in tree-sitter version)
        """
        self.tokenizer = tokenizer

        # Lazily initialize parsers
        self._parsers = {}

    def _get_parser(self, language: str):
        """
        Get or create parser for the given language

        Args:
            language: Language identifier ('python', 'java', 'c_sharp')

        Returns:
            Parser instance
        """
        if language not in self._parsers:
            self._parsers[language] = get_parser(language)

        return self._parsers[language]

    def detect_language(self, code: str) -> str:
        """
        Detect the programming language of the code

        Args:
            code: Source code string

        Returns:
            Language identifier: 'python', 'java', 'c_sharp'
        """
        # Check for language-specific markers (order matters - check C# before Java)
        # C# and Java share some markers, so check C# first
        for lang in ['c_sharp', 'java', 'python']:
            markers = self.LANGUAGE_MARKERS.get(lang, [])
            for marker in markers:
                if marker in code:
                    return lang

        # Default to Python
        return 'python'

    def segment_code(self, code: str, language: Optional[str] = None) -> List[CodeChunk]:
        """
        Segment code into semantic chunks using tree-sitter

        Args:
            code: Source code string
            language: Optional language override (auto-detected if not provided)
                     Accepts: 'python', 'java', 'csharp' or 'c_sharp'

        Returns:
            List of CodeChunk objects representing code segments
        """
        if not code or not code.strip():
            return []

        # Detect language if not provided
        if language is None:
            language = self.detect_language(code)
        
        language_map = {
            'csharp': 'c_sharp',
            'c_sharp': 'c_sharp',
            'python': 'python',
            'java': 'java'
        }
        language = language_map.get(language, language)

        # Get parser for the language
        parser = self._get_parser(language)

        # Parse the code (no silent fallback)
        code_bytes = code.encode("utf-8", errors="replace")
        tree = parser.parse(code_bytes)
        root = tree.root_node

        # Extract semantic chunks (imports/classes/functions/methods)
        semantic_chunks: List[CodeChunk] = []
        self._extract_chunks(root, code, code_bytes, language, semantic_chunks)

        # Fill uncovered regions with MODULE_CODE chunks so every non-empty line
        # deterministically maps to some chunk (no giant “cover-all” chunk).
        chunks = self._fill_gaps_with_module_chunks(code, code_bytes, semantic_chunks)

        # Line numbers for semantic chunks are already set by tree-sitter.
        self._calculate_line_numbers(chunks, code)
        return chunks

    def _extract_chunks(
        self,
        node: Node,
        code: str,
        code_bytes: bytes,
        language: str,
        chunks: List[CodeChunk],
        parent_class: Optional[str] = None,
    ):
        """
        Extract code chunks using iterative DFS to avoid deep recursion
        """
        node_types = self.NODE_TYPES.get(language, {})

        # stack holds tuples: (node, parent_class)
        stack = [(node, parent_class)]

        while stack:
            current, current_parent = stack.pop()
            node_type = current.type

            # Imports
            if node_type in node_types.get('import', []):
                chunks.append(self._create_chunk(current, code, code_bytes, ChunkType.IMPORT, "import"))
                continue

            # Classes
            if node_type in node_types.get('class', []):
                class_name = self._get_identifier(current, code_bytes, language)
                chunks.append(self._create_chunk(current, code, code_bytes, ChunkType.CLASS, class_name))
                # Process children with this class as parent (reverse to preserve order)
                for child in reversed(current.children):
                    stack.append((child, class_name))
                continue

            # Functions / methods
            if node_type in node_types.get('function', []):
                func_name = self._get_identifier(current, code_bytes, language)
                if current_parent:
                    chunks.append(self._create_chunk(current, code, code_bytes, ChunkType.METHOD, func_name, current_parent))
                else:
                    chunks.append(self._create_chunk(current, code, code_bytes, ChunkType.FUNCTION, func_name))
                # Do not descend into function bodies
                continue

            # Generic children traversal
            for child in reversed(current.children):
                stack.append((child, current_parent))

    def _create_chunk(
        self,
        node: Node,
        code: str,
        code_bytes: bytes,
        chunk_type: ChunkType,
        name: str,
        parent_name: Optional[str] = None,
    ) -> CodeChunk:
        """
        Create a CodeChunk from a syntax tree node

        Args:
            node: Syntax tree node
            code: Source code string
            chunk_type: Type of chunk
            name: Name of the chunk (function/class name)
            parent_name: Parent class name (for methods)

        Returns:
            CodeChunk object
        """
        start = node.start_byte
        end = node.end_byte
        text = code_bytes[start:end].decode("utf-8", errors="replace")

        sig_start = node.start_point[0] + 1
        sig_end = sig_start
        if chunk_type in (ChunkType.CLASS, ChunkType.FUNCTION, ChunkType.METHOD):
            signature_start_line = sig_start
            signature_end_line = sig_end
        else:
            signature_start_line = None
            signature_end_line = None

        return CodeChunk(
            chunk_type=chunk_type,
            start=start,
            end=end,
            text=text,
            name=name,
            parent_name=parent_name,
            start_line=node.start_point[0] + 1,  # tree-sitter uses 0-indexed lines
            end_line=node.end_point[0] + 1,
            signature_start_line=signature_start_line,
            signature_end_line=signature_end_line,
        )

    def _get_identifier(self, node: Node, code_bytes: bytes, language: str) -> str:
        """
        Extract the identifier (name) from a node

        Args:
            node: Syntax tree node
            code: Source code string
            language: Programming language

        Returns:
            Identifier string
        """
        # Look for identifier child node
        for child in node.children:
            if child.type == 'identifier':
                return code_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")

        # Fallback: try to find any named child
        for child in node.named_children:
            if child.type == 'identifier':
                return code_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")

        return "unknown"

    def _compute_line_start_bytes(self, code_bytes: bytes) -> List[int]:
        """
        Return list of byte offsets for each line start (1-based lines -> index line-1).
        """
        starts = [0]
        for i, b in enumerate(code_bytes):
            if b == 0x0A:  # b'\\n'
                starts.append(i + 1)
        return starts

    def _byte_span_for_lines(self, line_starts: List[int], code_bytes: bytes, start_line: int, end_line: int) -> Tuple[int, int]:
        if start_line < 1:
            start_line = 1
        if end_line < start_line:
            end_line = start_line

        num_lines = len(line_starts)
        if num_lines == 0:
            return 0, len(code_bytes)

        start_idx = min(start_line - 1, num_lines - 1)
        start_b = line_starts[start_idx]

        # end_line is inclusive; end byte is start of next line, or EOF.
        if end_line >= num_lines:
            end_b = len(code_bytes)
        else:
            end_b = line_starts[end_line]
        return start_b, end_b

    def _fill_gaps_with_module_chunks(self, code: str, code_bytes: bytes, semantic_chunks: List[CodeChunk]) -> List[CodeChunk]:
        """
        Ensure chunk coverage for the full snippet by creating MODULE_CODE chunks
        for uncovered regions (typically comments/license blocks/loose statements).

        This is not a fallback: it deterministically partitions uncovered text so
        token-to-chunk mapping stays well-defined and debuggable.
        """
        lines = code.split("\n")
        num_lines = len(lines)

        # Mark which lines are covered by at least one semantic chunk.
        covered = [False] * (num_lines + 1)  # 1..num_lines
        for c in semantic_chunks:
            s = max(1, int(getattr(c, "start_line", 1) or 1))
            e = min(num_lines, int(getattr(c, "end_line", num_lines) or num_lines))
            for ln in range(s, e + 1):
                covered[ln] = True

        # Build module chunks for uncovered regions, but avoid creating pure-blank chunks.
        line_starts = self._compute_line_start_bytes(code_bytes)
        module_chunks: List[CodeChunk] = []

        def is_blank_range(sln: int, eln: int) -> bool:
            for ln in range(sln, eln + 1):
                if lines[ln - 1].strip():
                    return False
            return True

        def _rebuild_chunk_for_lines(chunk: CodeChunk, new_start_line: int, new_end_line: int) -> CodeChunk:
            start_b, end_b = self._byte_span_for_lines(line_starts, code_bytes, new_start_line, new_end_line)
            return CodeChunk(
                chunk_type=chunk.chunk_type,
                start=start_b,
                end=end_b,
                text=code_bytes[start_b:end_b].decode("utf-8", errors="replace"),
                name=getattr(chunk, "name", "") or "",
                parent_name=getattr(chunk, "parent_name", None),
                start_line=new_start_line,
                end_line=new_end_line,
                signature_start_line=getattr(chunk, "signature_start_line", None),
                signature_end_line=getattr(chunk, "signature_end_line", None),
            )

        i = 1
        while i <= num_lines:
            if covered[i]:
                i += 1
                continue

            j = i
            while j <= num_lines and not covered[j]:
                j += 1
            start_line, end_line = i, j - 1

            # If it's only whitespace/empty lines, merge into an adjacent chunk (module or semantic)
            # to avoid standalone blank chunks while keeping full line coverage.
            if is_blank_range(start_line, end_line):
                # 1) Merge into previous module chunk if contiguous.
                if module_chunks and module_chunks[-1].end_line == start_line - 1:
                    prev = module_chunks[-1]
                    module_chunks[-1] = _rebuild_chunk_for_lines(prev, prev.start_line, end_line)
                    i = j
                    continue

                # 2) Merge into previous semantic chunk if contiguous.
                prev_sem_idx = None
                prev_sem_end = -1
                for idx, c in enumerate(semantic_chunks):
                    if c.end_line == start_line - 1 and c.end_line > prev_sem_end:
                        prev_sem_idx = idx
                        prev_sem_end = c.end_line
                if prev_sem_idx is not None:
                    c = semantic_chunks[prev_sem_idx]
                    semantic_chunks[prev_sem_idx] = _rebuild_chunk_for_lines(c, c.start_line, end_line)
                    i = j
                    continue

                # 3) Merge into next semantic chunk if contiguous.
                next_sem_idx = None
                next_sem_start = 10**18
                for idx, c in enumerate(semantic_chunks):
                    if c.start_line == end_line + 1 and c.start_line < next_sem_start:
                        next_sem_idx = idx
                        next_sem_start = c.start_line
                if next_sem_idx is not None:
                    c = semantic_chunks[next_sem_idx]
                    semantic_chunks[next_sem_idx] = _rebuild_chunk_for_lines(c, start_line, c.end_line)
                    i = j
                    continue

                # 4) Last resort: create a MODULE_CODE chunk (should be rare).
                start_b, end_b = self._byte_span_for_lines(line_starts, code_bytes, start_line, end_line)
                module_chunks.append(
                    CodeChunk(
                        chunk_type=ChunkType.MODULE_CODE,
                        start=start_b,
                        end=end_b,
                        text=code_bytes[start_b:end_b].decode("utf-8", errors="replace"),
                        name="",
                        parent_name=None,
                        start_line=start_line,
                        end_line=end_line,
                        signature_start_line=None,
                        signature_end_line=None,
                    )
                )
                i = j
                continue

            start_b, end_b = self._byte_span_for_lines(line_starts, code_bytes, start_line, end_line)
            module_chunks.append(
                CodeChunk(
                    chunk_type=ChunkType.MODULE_CODE,
                    start=start_b,
                    end=end_b,
                    text=code_bytes[start_b:end_b].decode("utf-8", errors="replace"),
                    name="",
                    parent_name=None,
                    start_line=start_line,
                    end_line=end_line,
                    signature_start_line=None,
                    signature_end_line=None,
                )
            )
            i = j

        # Deterministic order: keep original semantic order, but place module chunks early so
        # semantic chunks can override line->chunk mapping on overlaps in downstream consumers.
        # (Module chunks are gap-only in most cases, so overlap is rare.)
        return module_chunks + semantic_chunks

    def _calculate_line_numbers(self, chunks: List[CodeChunk], code: str):
        """
        Line numbers are already set by tree-sitter's start_point and end_point
        Return chunks using the same public interface as the previous segmenter.
        """
        pass
