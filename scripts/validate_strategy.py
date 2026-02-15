#!/usr/bin/env python3
"""
Run full 4-step validation for a strategy.

Usage:
python scripts/validate_strategy.py --strategy donchian --data data.csv
python scripts/validate_strategy.py --strategy donchian --data data.csv --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.validation.full_validation import run_full_validation


def parse_args():
    p = argparse.ArgumentParser(description="Run 4-step strategy validation")
    p.add_argument("--strategy", required=True, help="Strategy name from registry")
    p.add_argument("--data", required=True, help="CSV data path with OHLC columns")
    p.add_argument("--train-years", type=float, default=4.0)
    p.add_argument("--retrain-days", type=int, default=30)
    p.add_argument("--bars-per-day", type=int, default=1)
    p.add_argument("--n-perms-is", type=int, default=1000, help="In-sample permutations")
    p.add_argument("--n-perms-wf", type=int, default=200, help="Walk-forward permutations")
    p.add_argument("--output-root", default="reports")
    p.add_argument("--quick", action="store_true", help="Quick mode: skip permutation tests")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    result = run_full_validation(
        strategy_name=args.strategy,
        data_path=args.data,
        n_perms_is=args.n_perms_is,
        n_perms_wf=args.n_perms_wf,
        train_years=args.train_years,
        retrain_days=args.retrain_days,
        bars_per_day=args.bars_per_day,
        output_root=args.output_root,
        quick=args.quick,
    )

    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    print(f"Strategy: {result['strategy_name']}")
    print(f"Verdict: {result['verdict']}")
    if "is_pvalue" in result:
        print(f"In-sample PF: {result['is_pf']:.4f} (p={result['is_pvalue']:.6f})")
        print(f"OOS PF: {result['oos_pf']:.4f} (p={result['wf_pvalue']:.6f})")
    else:
        print(f"In-sample PF: {result['is_pf']:.4f}")
    if result.get("failed_step"):
        print(f"Failed step: {result['failed_step']}")
    print(f"Report: {result['report_path']}")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
