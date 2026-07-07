"""End-to-end Pine translation pipeline."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

from .parser import PineParser
from .translator import PineTranslator
from .validator import TranslationValidator


class TranslationPipeline:
    """Translate raw Pine code into a Python strategy module."""

    def __init__(self, strategy_dir: str | Path = "src/strategies"):
        self.strategy_dir = Path(strategy_dir)
        self.strategy_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_module_name(strategy_name: str) -> str:
        module = re.sub(r"[^a-z0-9_]+", "_", strategy_name.lower()).strip("_")
        return module or "auto_strategy"

    def _save_strategy(self, strategy_name: str, code: str,
                       pine_code: str | None = None) -> str:
        module_name = self._safe_module_name(strategy_name)
        path = self.strategy_dir / f"{module_name}.py"
        path.write_text(code, encoding="utf-8")
        if pine_code:
            # Keep the original Pine next to the module so the UI can show it
            (self.strategy_dir / f"{module_name}.pine").write_text(
                pine_code, encoding="utf-8")
        return str(path)

    # Strong signals that pasted text is Python source, not Pine Script.
    _PY_SIGNALS = re.compile(
        r"^\s*(?:import|from)\s+\w+|^\s*def\s+\w+\s*\(|^\s*class\s+\w+\s*[(:]"
        r"|\bself\.\w|\bBaseStrategy\b", re.M)

    @classmethod
    def _reject_reason(cls, pine_code: str) -> str | None:
        """A single clear reason this input can't be translated, or None."""
        if not pine_code.strip():
            return "Nothing to translate — paste a Pine Script strategy first."
        if "//@version" not in pine_code and cls._PY_SIGNALS.search(pine_code):
            return ("This looks like Python, not Pine Script. Use "
                    "\"New Python strategy\" in the library to save it instead.")
        if not re.search(r"\bstrategy\s*\(", pine_code):
            if re.search(r"\b(?:indicator|study)\s*\(", pine_code):
                return ("This script declares indicator(), not strategy() — it plots "
                        "values but never places orders, so there is nothing to "
                        "backtest. Convert it: use strategy() and add "
                        "strategy.entry()/strategy.close() rules.")
            return ("No strategy() declaration found — paste a complete Pine "
                    "strategy (it must start with strategy(\"Name\", ...)).")
        if "strategy.entry" not in pine_code:
            if "strategy.order" in pine_code:
                return ("strategy.order() isn't supported yet — rewrite the entries "
                        "with strategy.entry() (exits via strategy.exit()/strategy.close()).")
            return ("No strategy.entry() calls found — this script would never "
                    "open a trade, so there is nothing to backtest.")
        return None

    def translate(self, pine_code: str, auto_save: bool = True) -> Dict:
        """
        Translate Pine code to Python strategy source.
        """
        result = {
            "success": False,
            "strategy_name": None,
            "python_code": None,
            "python_file": None,
            "issues": [],
            "warnings": [],
            "manual_review_needed": False,
        }

        reason = self._reject_reason(pine_code)
        if reason:
            result["issues"].append(reason)
            return result

        try:
            parser = PineParser(pine_code)
            parsed = parser.parse()
            result["strategy_name"] = parsed.name

            translator = PineTranslator(parsed)
            code = translator.generate_python_code()
            result["python_code"] = code

            validation = TranslationValidator(code).validate()
            result["issues"] = validation["issues"]
            # translator warnings describe every Pine construct we degraded
            result["warnings"] = list(dict.fromkeys(
                translator.warnings + validation["warnings"]))
            result["manual_review_needed"] = not validation["valid"]

            result["success"] = len(validation["issues"]) == 0
            # Never write a broken strategy into the user's library.
            if auto_save and result["success"]:
                file_path = self._save_strategy(parsed.name, code, pine_code)
                result["python_file"] = file_path
            if result["warnings"]:
                result["manual_review_needed"] = True

        except Exception as exc:
            result["issues"].append(f"Translation failed: {exc}")
            result["success"] = False
            result["manual_review_needed"] = True

        return result
