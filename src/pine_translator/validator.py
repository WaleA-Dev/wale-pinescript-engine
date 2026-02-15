"""Validation checks for generated strategy code."""

from __future__ import annotations

import ast
import re
from typing import Dict, List


class TranslationValidator:
    """Simple static validator for generated Python code."""

    def __init__(self, python_code: str):
        self.python_code = python_code

    def validate(self) -> Dict:
        issues: List[str] = []
        warnings: List[str] = []

        if not self.python_code.strip():
            return {"valid": False, "issues": ["Generated code is empty"], "warnings": warnings}

        try:
            ast.parse(self.python_code)
        except SyntaxError as exc:
            issues.append(f"Syntax error: {exc}")

        if "class " not in self.python_code:
            issues.append("Missing class definition")
        if "def generate_signals" not in self.python_code:
            issues.append("Missing generate_signals method")
        if "BaseStrategy" not in self.python_code:
            issues.append("Generated class does not inherit BaseStrategy")

        todo_count = len(re.findall(r"TODO:", self.python_code))
        if todo_count > 0:
            warnings.append(f"{todo_count} untranslated function(s) marked TODO")

        valid = len(issues) == 0 and todo_count == 0
        return {"valid": valid, "issues": issues, "warnings": warnings}
