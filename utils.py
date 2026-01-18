import ast
from typing import Callable, Any, Dict, List, Set,  Optional
import json
import re
import random

FORBIDDEN = {
    "exec", "eval", "compile", "globals", "locals", "open",
    "type", "getattr", "setattr", "delattr", "vars", "dir"
}

FORBIDDEN_IMPORTS = {
    "os", "sys", "subprocess", "socket", "requests", "urllib", "urllib3",
    "http", "ftplib", "smtplib", "telnetlib", "pathlib", "shutil",
    "importlib", "pkgutil", "zipimport", "runpy", "code", "codeop",
    "pty", "tty", "atexit", "signal", "ctypes", "multiprocessing",
}

ALLOWED_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, 
    "enumerate": enumerate, "filter": filter, "float": float, "int": int,
    "len": len, "list": list, "map": map, "max": max, "min": min,
    "range": range, "round": round, "set": set, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "True": True, "False": False, "None": None,
    "print": print, "isinstance": isinstance, "issubclass": issubclass,
    "__import__": __import__,  # Needed for import statements to work
}


class SafeValidator(ast.NodeVisitor):
    def visit_Import(self, node):
        for alias in node.names:
            if alias.name in FORBIDDEN_IMPORTS:
                raise ValueError(f"Import '{alias.name}' not allowed")
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module and node.module.split('.')[0] in FORBIDDEN_IMPORTS:
            raise ValueError(f"Import from '{node.module}' not allowed")
        self.generic_visit(node)
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN:
            raise ValueError(f"Forbidden: {node.func.id}")
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        if node.attr.startswith("__"):
            raise ValueError(f"Dunder access blocked: {node.attr}")
        self.generic_visit(node)
    
    def visit_Name(self, node):
        if node.id in FORBIDDEN:
            raise ValueError(f"Forbidden: {node.id}")
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        raise ValueError("Class definitions not allowed")
    
    def visit_AsyncFunctionDef(self, node):
        raise ValueError("Async functions not allowed")


def compile_safe_function(func_str: str, max_len: int = 10000) -> Callable:
    """Compile a function string with security restrictions."""
    if not func_str.strip() or len(func_str) > max_len:
        raise ValueError("Invalid function string")
    
    tree = ast.parse(func_str)
    
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Exactly one function definition required")
    
    SafeValidator().visit(tree)
    
    namespace = {"__builtins__": ALLOWED_BUILTINS}
    exec(compile(tree, "<safe>", "exec"), namespace)
    
    return namespace[tree.body[0].name]

def random_selection(ds: Any, k: int = 100) -> List[Any]:
    """Select k random examples from a dataset using reservoir sampling."""
    random.seed(42)
    reservoir = []
    
    for i, ex in enumerate(ds):
        if i < k:
            reservoir.append(ex)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = ex
    
    return reservoir

def extract_json_from_text(pred_text: str) -> Optional[dict[str, Any]]:
    """
    Extract and parse JSON from LLM response text with multiple fallback strategies.
    
    Args:
        pred_text: Raw text from LLM that may contain JSON
        
    Returns:
        Parsed JSON as dict, or None if parsing fails
    """
    if not pred_text or not pred_text.strip():
        return None
    
    # Attempt 1: Parse as clean JSON
    try:
        return json.loads(pred_text)
    except json.JSONDecodeError:
        pass
    
    # Attempt 2: Remove markdown code blocks
    try:
        # Match ```json or ``` followed by content
        pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        match = re.search(pattern, pred_text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Attempt 3: Extract JSON object by finding balanced braces
    try:
        json_str = extract_balanced_json(pred_text, '{', '}')
        if json_str:
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Attempt 4: Extract JSON array by finding balanced brackets
    try:
        json_str = extract_balanced_json(pred_text, '[', ']')
        if json_str:
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Attempt 5: Try cleaning common issues
    try:
        cleaned = clean_json_string(pred_text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    return None

def extract_balanced_json(text: str, open_char: str, close_char: str) -> Optional[str]:
    """
    Extract JSON by finding balanced opening/closing characters.
    Handles nested structures correctly.
    """
    start = text.find(open_char)
    if start == -1:
        return None
    
    count = 0
    in_string = False
    escape = False
    
    for i in range(start, len(text)):
        char = text[i]
        
        # Handle string literals to ignore braces/brackets inside strings
        if char == '"' and not escape:
            in_string = not in_string
        elif char == '\\' and in_string:
            escape = not escape
            continue
        
        if not in_string:
            if char == open_char:
                count += 1
            elif char == close_char:
                count -= 1
                if count == 0:
                    return text[start:i+1]
        
        escape = False
    
    return None

def clean_json_string(text: str) -> str:
    """
    Clean common formatting issues in JSON strings.
    """
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove common prefixes like "Here's the JSON:" or "Output:"
    text = re.sub(r'^(?:Here\'s|Here is|Output:?|Result:?|JSON:?)\s*', '', text, flags=re.IGNORECASE)
    
    # Remove trailing text after JSON (greedy approach)
    # Find the last } or ] and cut everything after
    last_brace = max(text.rfind('}'), text.rfind(']'))
    if last_brace != -1:
        text = text[:last_brace + 1]
    
    return text