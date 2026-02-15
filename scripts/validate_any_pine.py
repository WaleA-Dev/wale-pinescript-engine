#!/usr/bin/env python3
"""
Validate ANY PineScript strategy.

Usage:
    python scripts/validate_any_pine.py --pine my_strategy.pine --data bitcoin.csv
    python scripts/validate_any_pine.py --pine-stdin --data bitcoin.csv
    python scripts/validate_any_pine.py --interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pine_translator.pipeline import TranslationPipeline
from src.validation.full_validation import run_full_validation


def interactive_pine_input() -> str:
    print("\n" + "=" * 60)
    print("INTERACTIVE PINE INPUT")
    print("=" * 60)
    print("Paste PineScript code below, then press Ctrl+Z and Enter on Windows.")
    lines = []
    try:
        while True:
            lines.append(input())
    except EOFError:
        pass
    return "\n".join(lines)


def estimate_runtime(n_is: int, n_wf: int) -> int:
    # rough estimate in minutes
    return int((n_is * 0.5 + n_wf * 2.0) / 60.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Translate and validate any Pine strategy")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--pine", type=str, help="Path to .pine file")
    input_group.add_argument("--pine-stdin", action="store_true", help="Read Pine from stdin")
    input_group.add_argument("--interactive", action="store_true", help="Interactive Pine input")

    parser.add_argument("--data", type=str, help="OHLC CSV data path")
    parser.add_argument("--skip-validation", action="store_true", help="Translate only")
    parser.add_argument("--n-perms-is", type=int, default=1000)
    parser.add_argument("--n-perms-wf", type=int, default=200)
    parser.add_argument("--train-years", type=float, default=4.0)
    parser.add_argument("--retrain-days", type=int, default=30)
    parser.add_argument("--bars-per-day", type=int, default=24)
    parser.add_argument("--output-root", default="reports")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.pine:
        pine_code = Path(args.pine).read_text(encoding="utf-8")
    elif args.pine_stdin:
        print("Paste PineScript code, then Ctrl+Z + Enter:")
        pine_code = sys.stdin.read()
    else:
        pine_code = interactive_pine_input()

    print("\n" + "=" * 60)
    print("STEP 1: TRANSLATING PINE TO PYTHON")
    print("=" * 60)
    pipeline = TranslationPipeline()
    trans = pipeline.translate(pine_code, auto_save=True)

    if not trans["success"]:
        print("\nTRANSLATION FAILED")
        for issue in trans["issues"]:
            print(f"- {issue}")
        return 1

    print(f"Translation successful: {trans['strategy_name']}")
    print(f"Generated file: {trans['python_file']}")
    if trans["warnings"]:
        print("Warnings:")
        for w in trans["warnings"]:
            print(f"- {w}")
    if trans["manual_review_needed"]:
        print("\nMANUAL REVIEW REQUIRED")
        print(f"Generated code saved to: {trans['python_file']}")
        print("Please review TODO items before running validation.")
        return 0

    if args.skip_validation:
        print("Skipping validation (--skip-validation)")
        return 0

    if not args.data:
        print("No --data provided. Translation complete but validation skipped.")
        return 0

    print("\n" + "=" * 60)
    print("STEP 2: RUNNING 4-STEP VALIDATION")
    print("=" * 60)
    print(f"Estimated runtime: ~{estimate_runtime(args.n_perms_is, args.n_perms_wf)} minutes")

    result = run_full_validation(
        strategy_name=trans["strategy_name"],
        data_path=args.data,
        n_perms_is=args.n_perms_is,
        n_perms_wf=args.n_perms_wf,
        train_years=args.train_years,
        retrain_days=args.retrain_days,
        bars_per_day=args.bars_per_day,
        output_root=args.output_root,
    )

    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    if result["verdict"] == "VALIDATED":
        print("VALIDATED")
        print(f"IS PF: {result['is_pf']:.2f}, p={result['is_pvalue']:.4f}")
        print(f"OOS PF: {result['oos_pf']:.2f}, p={result['wf_pvalue']:.4f}")
    elif result["verdict"] == "OVERFIT":
        print("OVERFIT")
        print(f"Failed step: {result['failed_step']}")
        print(f"p-value: {result['failed_pvalue']:.4f}" if result["failed_pvalue"] is not None else "")
    else:
        print("POOR")
        print(f"OOS PF: {result['oos_pf']:.2f}")
    print(f"Report: {result['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
