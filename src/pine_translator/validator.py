"""Validation for generated strategy code: syntax check + live smoke run."""

from __future__ import annotations

import ast
from typing import Dict, List

import numpy as np
import pandas as pd


def _synthetic_df(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    steps = rng.normal(0.0005, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(steps))
    spread = np.abs(rng.normal(0, 0.004, n)) * close
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol = rng.integers(1000, 100000, n).astype(float)
    idx = pd.date_range("2022-01-03", periods=n, freq="D")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


class TranslationValidator:
    """Static + dynamic validation of transpiled strategy code."""

    def __init__(self, python_code: str):
        self.python_code = python_code

    def validate(self) -> Dict:
        issues: List[str] = []
        warnings: List[str] = []

        if not self.python_code.strip():
            return {"valid": False, "issues": ["Generated code is empty"],
                    "warnings": warnings}

        try:
            ast.parse(self.python_code)
        except SyntaxError as exc:
            issues.append(f"Generated code has a syntax error: {exc}")
            return {"valid": False, "issues": issues, "warnings": warnings}

        if "class " not in self.python_code:
            issues.append("Missing class definition")
        if "PineStrategy" not in self.python_code:
            issues.append("Generated class does not inherit PineStrategy")
        if issues:
            return {"valid": False, "issues": issues, "warnings": warnings}

        # Dynamic smoke test: exec + run on synthetic data
        try:
            from src.strategies.pine_base import PineStrategy
            namespace: Dict = {}
            exec(compile(self.python_code, "<translated>", "exec"), namespace)
            cls = None
            for obj in namespace.values():
                if (isinstance(obj, type) and issubclass(obj, PineStrategy)
                        and obj is not PineStrategy):
                    cls = obj
                    break
            if cls is None:
                issues.append("No PineStrategy subclass found in generated code")
            else:
                strat = cls()
                result = strat.run(_synthetic_df())
                if not isinstance(result, dict) or "metrics" not in result:
                    issues.append("Smoke run did not return a valid result")
                elif result.get("bar_errors", 0) > 0:
                    issues.append(
                        f"Strategy raised errors on {result['bar_errors']} bars "
                        f"(first: {result.get('first_error')})")
        except Exception as exc:
            issues.append(f"Smoke run failed: {type(exc).__name__}: {exc}")

        return {"valid": len(issues) == 0, "issues": issues, "warnings": warnings}
