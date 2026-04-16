"""
CCEval-specific utilities for proper evaluation.

This module implements the correct truncation logic from CCEval benchmark.
"""

import re
import ast
from typing import Optional

# Identifier regex from CCEval
IDENTIFIER_REGEX = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')


def get_bracket_lang_statement(completion: str) -> str:
    """
    For bracket languages (Java, C#, TypeScript), truncate to first statement.
    Stops at first occurrence of ;, }, or {
    
    Args:
        completion: Generated code
        
    Returns:
        Truncated code
    """
    end_idx = None
    for i in range(len(completion)):
        if completion[i] in [";", "}", "{"]:
            end_idx = i
            break
    return completion[:end_idx + 1] if end_idx is not None else completion


def get_python_one_statement(prompt: str, completion: str, use_ast: bool = True) -> str:
    """
    For Python, truncate to first complete statement.
    Stops at first newline that results in valid syntax.
    
    Args:
        prompt: The prompt/context before completion
        completion: Generated code
        use_ast: Whether to use AST parsing (more accurate but slower)
        
    Returns:
        Truncated code
    """
    if not completion:
        return completion
    
    # Simple version: stop at first newline
    if not use_ast:
        newline_idx = completion.find('\n')
        if newline_idx != -1:
            return completion[:newline_idx].rstrip()
        return completion
    
    # AST-based version (more accurate)
    try:
        for i in range(len(completion)):
            if i + 1 < len(completion) and completion[i] == '\n':
                # Try to parse up to this point
                code = prompt + completion[:i + 1]
                try:
                    ast.parse(code)
                    # If parsing succeeds, this is a complete statement
                    return completion[:i + 1].rstrip()
                except SyntaxError:
                    # Not yet a complete statement, continue
                    continue
    except Exception:
        pass
    
    # Fallback: stop at first newline
    newline_idx = completion.find('\n')
    if newline_idx != -1:
        return completion[:newline_idx].rstrip()
    
    return completion


def postprocess_code_lines(
    prompt: str,
    completion: str,
    language: str,
    use_ast: bool = False
) -> str:
    """
    Post-process generated code according to CCEval rules.
    
    This truncates the generated code to the first complete statement,
    which is the correct way to evaluate line completion tasks.
    
    Args:
        prompt: The prompt/context before completion
        completion: Generated code
        language: Programming language (python, java, csharp, typescript)
        use_ast: Whether to use AST parsing for Python (more accurate but slower)
        
    Returns:
        Truncated code
    """
    if not completion:
        return completion
    
    try:
        language = language.lower()
        
        if language in ["java", "csharp", "typescript"]:
            return get_bracket_lang_statement(completion)
        elif language == "python":
            return get_python_one_statement(prompt, completion, use_ast=use_ast)
        else:
            # Unknown language, use simple newline truncation
            newline_idx = completion.find('\n')
            if newline_idx != -1:
                return completion[:newline_idx].rstrip()
            return completion
            
    except Exception as e:
        # On any error, return original completion
        return completion


def remove_comments_simple(code: str) -> str:
    """
    Remove comments from code (simple version).
    
    Args:
        code: Source code
        
    Returns:
        Code with comments removed
    """
    # Remove Python/Shell style comments
    code = re.sub(r'#.*', '', code)
    # Remove C/C++/Java style comments
    code = re.sub(r'//.*', '', code)
    return code


def extract_identifiers_simple(code: str, language: Optional[str] = None) -> list:
    """
    Extract identifiers from code (simple version without keyword filtering).
    
    Args:
        code: Source code
        language: Programming language (optional, for keyword filtering)
        
    Returns:
        List of identifiers
    """
    # Remove strings first
    string_pattern = r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\''
    code_without_strings = re.sub(string_pattern, '', code)
    
    # Find all identifiers
    identifiers = IDENTIFIER_REGEX.findall(code_without_strings)
    
    # TODO: Filter out language keywords if needed
    # For now, return all identifiers
    return identifiers


# Example usage and testing
if __name__ == "__main__":
    # Test Python truncation
    prompt = "def foo():\n    x = "
    completion = "1 + 2\n    y = 3\n    return x + y"
    
    truncated = postprocess_code_lines(prompt, completion, "python")
    print(f"Original: {repr(completion)}")
    print(f"Truncated: {repr(truncated)}")
    
    # Test Java truncation
    completion_java = "int x = 5; int y = 10; return x + y;"
    truncated_java = postprocess_code_lines("", completion_java, "java")
    print(f"\nJava Original: {repr(completion_java)}")
    print(f"Java Truncated: {repr(truncated_java)}")

