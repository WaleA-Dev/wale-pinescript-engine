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

    def _save_strategy(self, strategy_name: str, code: str) -> str:
        module_name = self._safe_module_name(strategy_name)
        path = self.strategy_dir / f"{module_name}.py"
        path.write_text(code, encoding="utf-8")
        return str(path)

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

        try:
            parser = PineParser(pine_code)
            parsed = parser.parse()
            result["strategy_name"] = parsed.name

            translator = PineTranslator(parsed)
            code = translator.generate_python_code()
            result["python_code"] = code

            validation = TranslationValidator(code).validate()
            result["issues"] = validation["issues"]
            result["warnings"] = validation["warnings"]
            result["manual_review_needed"] = not validation["valid"]

            if auto_save:
                file_path = self._save_strategy(parsed.name, code)
                result["python_file"] = file_path

            result["success"] = len(validation["issues"]) == 0
            if result["warnings"]:
                result["manual_review_needed"] = True

        except Exception as exc:
            result["issues"].append(f"Translation failed: {exc}")
            result["success"] = False
            result["manual_review_needed"] = True

        return result
